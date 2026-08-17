/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package controller

import (
	"context"
	"errors"
	"fmt"
	"maps"
	"strconv"

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	commoncontroller "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/provideroverride"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	corev1 "k8s.io/api/core/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

type groveProviderProgramIdentity struct {
	Spec        any               `json:"spec"`
	Labels      map[string]string `json:"labels,omitempty"`
	Annotations map[string]string `json:"annotations,omitempty"`
}

// reconcileGroveProviderProgram validates and applies one complete Grove
// provider program. dgd and desired must not be nil.
func (r *groveWorkloadsReconciler) reconcileGroveProviderProgram(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	desired *unstructured.Unstructured,
) (*grovev1alpha1.PodCliqueSet, error) {
	// Attach stable ownership and the rendered-program hash before applying.
	if err := ctrl.SetControllerReference(dgd, desired, r.syncer.Scheme()); err != nil {
		return nil, fmt.Errorf("set PodCliqueSet controller reference: %w", err)
	}
	desiredHash, err := hashGroveProviderProgram(desired)
	if err != nil {
		return nil, fmt.Errorf("hash PodCliqueSet provider program: %w", err)
	}
	setAnnotation(desired, commoncontroller.NvidiaAnnotationHashKey, desiredHash)

	// Observe the current backing object once from the informer cache.
	key := types.NamespacedName{Name: desired.GetName(), Namespace: desired.GetNamespace()}
	live, liveExists, err := r.getGroveProviderProgram(ctx, key)
	if err != nil {
		return nil, err
	}

	// Return unchanged before making provider availability a reconciliation dependency.
	if liveExists && groveProviderProgramIsCurrent(live, desired, desiredHash) {
		return live, nil
	}

	// Validate the exact program immediately before mutating the workload.
	if err := provideroverride.DryRunGroveProgram(ctx, r.syncer.Client, desired); err != nil {
		return nil, failWorkloadProgram(
			groveProviderPreflightFailureReason(err, liveExists),
			fmt.Errorf("validate Grove provider program: %w", err),
		)
	}

	// Apply the exact validated payload with one durable field manager.
	if err := r.syncer.Apply(
		ctx,
		client.ApplyConfigurationFromUnstructured(desired),
		client.FieldOwner(provideroverride.GroveFieldOwner),
		client.ForceOwnership,
	); err != nil {
		return nil, fmt.Errorf("apply PodCliqueSet provider program: %w", err)
	}
	recordGroveProviderProgramMutation(r, desired, key, liveExists)

	// Capture the program that is already authoritative before recording bookkeeping metadata.
	synced, err := typedPodCliqueSet(desired)
	if err != nil {
		return nil, err
	}

	// Preserve the server-returned generation marker without a read-after-write.
	original := desired.DeepCopy()
	setAnnotation(desired, commoncontroller.NvidiaAnnotationGenerationKey, strconv.FormatInt(desired.GetGeneration(), 10))
	if err := r.syncer.Patch(
		ctx,
		desired,
		client.MergeFrom(original),
		client.FieldOwner(provideroverride.GroveFieldOwner),
	); err != nil {
		return synced, fmt.Errorf("record applied PodCliqueSet provider program: %w", err)
	}

	// Return the typed readiness view from the applied program.
	return synced, nil
}

// rejectGroveProviderProgram classifies a locally rejected envelope without
// mutating an existing backing workload. desired and programErr must not be nil.
func (r *groveWorkloadsReconciler) rejectGroveProviderProgram(
	ctx context.Context,
	desired *grovev1alpha1.PodCliqueSet,
	programErr error,
) error {
	// Observe whether a backing object exists before classifying the failure.
	key := types.NamespacedName{Name: desired.Name, Namespace: desired.Namespace}
	_, liveExists, err := r.getGroveProviderProgram(ctx, key)
	if err != nil {
		return err
	}

	// Return before any mutation so a live provider program remains untouched.
	return failWorkloadProgram(
		localGroveProviderProgramFailureReason(liveExists),
		fmt.Errorf("validate Grove provider program: %w", programErr),
	)
}

// getGroveProviderProgram returns the cache-backed workload and an explicit
// presence bit for one provider-program key.
func (r *groveWorkloadsReconciler) getGroveProviderProgram(
	ctx context.Context,
	key types.NamespacedName,
) (*grovev1alpha1.PodCliqueSet, bool, error) {
	// Distinguish an absent workload from a cache read failure.
	live := &grovev1alpha1.PodCliqueSet{}
	if err := r.syncer.Get(ctx, key, live); err != nil {
		if apierrors.IsNotFound(err) {
			return nil, false, nil
		}
		return nil, false, fmt.Errorf("get PodCliqueSet provider program %s: %w", key, err)
	}
	return live, true, nil
}

// localGroveProviderProgramFailureReason classifies an envelope rejected before
// provider preflight.
func localGroveProviderProgramFailureReason(liveExists bool) Reason {
	// Without applied history, an absent backing object is simply unavailable.
	if !liveExists {
		return reasonProviderProgramUnavailable
	}
	return reasonProviderOverrideInvalid
}

// groveProviderPreflightFailureReason classifies an API-server dry-run failure.
func groveProviderPreflightFailureReason(err error, liveExists bool) Reason {
	// Without applied history, an absent backing object is simply unavailable.
	if !liveExists {
		return reasonProviderProgramUnavailable
	}

	// Classify typed rejections and controller-runtime's detail-free webhook denial as invalid.
	if apierrors.IsInvalid(err) || apierrors.IsBadRequest(err) {
		return reasonProviderOverrideInvalid
	}
	var status apierrors.APIStatus
	if apierrors.IsForbidden(err) && errors.As(err, &status) && status.Status().Details == nil {
		return reasonProviderOverrideInvalid
	}
	return reasonProviderValidationUnavailable
}

// groveProviderProgramIsCurrent verifies the complete operator-owned live
// identity without interpreting opaque provider fields. live and desired must
// not be nil.
func groveProviderProgramIsCurrent(
	live *grovev1alpha1.PodCliqueSet,
	desired *unstructured.Unstructured,
	desiredHash string,
) bool {
	// Generation covers spec drift, while the stored hash covers desired-program changes.
	annotations := live.GetAnnotations()
	if annotations[commoncontroller.NvidiaAnnotationHashKey] != desiredHash ||
		annotations[commoncontroller.NvidiaAnnotationGenerationKey] != strconv.FormatInt(live.Generation, 10) {
		return false
	}

	// Metadata generation is not tracked by Kubernetes, so compare every desired key directly.
	if !containsProviderProgramMetadata(live.GetLabels(), desired.GetLabels()) ||
		!containsProviderProgramMetadata(annotations, desired.GetAnnotations()) {
		return false
	}

	// Restore deletion, replacement, or mutation of the DGD controller reference.
	return apiequality.Semantic.DeepEqual(
		metav1.GetControllerOfNoCopy(live),
		metav1.GetControllerOfNoCopy(desired),
	)
}

func containsProviderProgramMetadata(live, desired map[string]string) bool {
	// Ignore unrelated live metadata while requiring every rendered key and value.
	for key, value := range desired {
		if live[key] != value {
			return false
		}
	}
	return true
}

func hashGroveProviderProgram(desired *unstructured.Unstructured) (string, error) {
	// Include every rendered field whose change must trigger a mutating apply.
	spec, found, err := unstructured.NestedFieldNoCopy(desired.Object, "spec")
	if err != nil {
		return "", fmt.Errorf("read PodCliqueSet spec: %w", err)
	}
	if !found {
		return "", fmt.Errorf("PodCliqueSet spec is missing")
	}

	// Exclude only reconciliation bookkeeping from the rendered metadata identity.
	annotations := maps.Clone(desired.GetAnnotations())
	delete(annotations, commoncontroller.NvidiaAnnotationHashKey)
	delete(annotations, commoncontroller.NvidiaAnnotationGenerationKey)
	if len(annotations) == 0 {
		annotations = nil
	}
	labels := maps.Clone(desired.GetLabels())
	if len(labels) == 0 {
		labels = nil
	}

	return commoncontroller.GetResourceHash(groveProviderProgramIdentity{
		Spec:        spec,
		Labels:      labels,
		Annotations: annotations,
	})
}

func setAnnotation(object metav1.Object, key, value string) {
	// Allocate metadata lazily while preserving unrelated annotations.
	annotations := object.GetAnnotations()
	if annotations == nil {
		annotations = make(map[string]string)
	}
	annotations[key] = value
	object.SetAnnotations(annotations)
}

func typedPodCliqueSet(object *unstructured.Unstructured) (*grovev1alpha1.PodCliqueSet, error) {
	// Convert only after provider validation and application have completed.
	synced := &grovev1alpha1.PodCliqueSet{}
	if err := runtime.DefaultUnstructuredConverter.FromUnstructured(object.Object, synced); err != nil {
		return nil, fmt.Errorf("convert applied PodCliqueSet %s/%s: %w", object.GetNamespace(), object.GetName(), err)
	}
	return synced, nil
}

func recordGroveProviderProgramMutation(
	r *groveWorkloadsReconciler,
	desired *unstructured.Unstructured,
	key types.NamespacedName,
	updated bool,
) {
	// Describe the successful mutation edge using the pre-apply existence observation.
	eventReason := "CreatePodCliqueSet"
	eventAction := "Create"
	eventMessage := "Created PodCliqueSet %s"

	// Switch the event vocabulary when reconciliation updated an existing workload.
	if updated {
		eventReason = "UpdatePodCliqueSet"
		eventAction = "Update"
		eventMessage = "Updated PodCliqueSet %s"
	}
	r.syncer.GetRecorder().Eventf(desired, nil, corev1.EventTypeNormal, eventReason, eventAction, eventMessage, key)
}
