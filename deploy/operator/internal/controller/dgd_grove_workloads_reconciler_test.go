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
	"net/http"
	"testing"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	commoncontroller "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/provideroverride"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/tools/events"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	"sigs.k8s.io/controller-runtime/pkg/client/interceptor"
)

func TestGroveWorkloadsReconciler_EvaluatesReadinessOnce(t *testing.T) {
	tests := []struct {
		name                   string
		currentOverride        bool
		previouslyManagedBySSA bool
		wantApplyCalls         int
	}{
		{name: "typed pathway", wantApplyCalls: 0},
		{name: "current provider override", currentOverride: true, wantApplyCalls: 2},
		{name: "removed provider override", previouslyManagedBySSA: true, wantApplyCalls: 2},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Log("Build a Grove DGD for the selected reconciliation pathway")
			dgd := betaDGD(t, &nvidiacomv1alpha1.DynamoGraphDeployment{
				ObjectMeta: metav1.ObjectMeta{Name: "graph", Namespace: "default"},
				Spec: nvidiacomv1alpha1.DynamoGraphDeploymentSpec{
					BackendFramework: "vllm",
					Services: map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
						"frontend": {
							ComponentType: consts.ComponentTypeFrontend,
							Replicas:      ptr.To(int32(1)),
						},
					},
				},
			})
			if tt.currentOverride {
				dgd.Spec.ProviderOverride = &nvidiacomv1beta1.ProviderOverride{
					APIVersion: provideroverride.GroveAPIVersion,
					Target:     provideroverride.TargetPodCliqueSet,
					Value: apiextensionsv1.JSON{Raw: []byte(
						`{"spec":{"template":{"topologyConstraint":{"topologyName":"cluster","pack":{"required":"rack"}}}}}`,
					)},
				}
			}

			t.Log("Expose one ready PodClique and any durable SSA pathway marker")
			podClique := &grovev1alpha1.PodClique{
				ObjectMeta: metav1.ObjectMeta{
					Name:       "graph-0-frontend",
					Namespace:  "default",
					Generation: 1,
				},
				Spec: grovev1alpha1.PodCliqueSpec{Replicas: 1},
				Status: grovev1alpha1.PodCliqueStatus{
					Replicas:           1,
					ReadyReplicas:      1,
					UpdatedReplicas:    1,
					ScheduledReplicas:  1,
					ObservedGeneration: ptr.To(int64(1)),
				},
			}
			objects := []client.Object{dgd, podClique}
			if tt.previouslyManagedBySSA {
				objects = append(objects, &grovev1alpha1.PodCliqueSet{ObjectMeta: metav1.ObjectMeta{
					Name:      "graph",
					Namespace: "default",
				}})
			}

			t.Log("Wire the Grove program with recording scale and apply clients")
			podCliqueReads := 0
			applyCalls := 0
			scaleClient := &recordingGroveScaleClient{}
			kubeClient := fake.NewClientBuilder().
				WithScheme(newDynamoGraphDeploymentControllerTestScheme(t)).
				WithObjects(objects...).
				WithStatusSubresource(dgd, podClique).
				WithInterceptorFuncs(interceptor.Funcs{
					Get: func(
						ctx context.Context,
						reader client.WithWatch,
						key client.ObjectKey,
						object client.Object,
						options ...client.GetOption,
					) error {
						if _, ok := object.(*grovev1alpha1.PodClique); ok {
							require.Len(t, scaleClient.updates, 1, "readiness must be observed after scaling")
							podCliqueReads++
						}
						getErr := reader.Get(ctx, key, object, options...)
						if getErr == nil && tt.previouslyManagedBySSA {
							if podCliqueSet, ok := object.(*grovev1alpha1.PodCliqueSet); ok {
								podCliqueSet.ManagedFields = []metav1.ManagedFieldsEntry{{
									Manager: provideroverride.GroveFieldOwner,
								}}
							}
						}
						return getErr
					},
					Apply: func(
						ctx context.Context,
						writer client.WithWatch,
						object runtime.ApplyConfiguration,
						options ...client.ApplyOption,
					) error {
						applyCalls++
						return writer.Apply(ctx, object, options...)
					},
				}).
				Build()
			reconciler := &DynamoGraphDeploymentReconciler{
				Client:        kubeClient,
				Config:        &configv1alpha1.OperatorConfiguration{},
				Recorder:      events.NewFakeRecorder(10),
				RuntimeConfig: &commoncontroller.RuntimeConfig{},
				ScaleClient:   scaleClient,
				DockerSecretRetriever: &mockDockerSecretRetriever{
					GetSecretsFunc: func(string, string) ([]string, error) {
						return nil, nil
					},
				},
			}

			t.Log("Reconcile the complete Grove workload program")
			result, err := reconciler.newGroveProgram().workloads.Reconcile(
				context.Background(),
				dgd,
				nil,
				nil,
			)

			t.Log("Verify pathway selection and one readiness observation after scaling")
			require.NoError(t, err)
			assert.Equal(t, nvidiacomv1beta1.DGDStateSuccessful, result.State)
			assert.Equal(t, tt.wantApplyCalls, applyCalls)
			assert.Equal(t, 1, podCliqueReads)
		})
	}
}

func TestGroveProviderOverrideReconcileDryRunsBeforeApply(t *testing.T) {
	t.Log("Build a future-aware unstructured Grove provider program")
	scheme := newDynamoGraphDeploymentControllerTestScheme(t)
	dgd := &nvidiacomv1beta1.DynamoGraphDeployment{
		TypeMeta: metav1.TypeMeta{
			APIVersion: nvidiacomv1beta1.GroupVersion.String(),
			Kind:       "DynamoGraphDeployment",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name:      "graph",
			Namespace: "default",
			UID:       types.UID("graph-uid"),
		},
	}
	desired := &unstructured.Unstructured{Object: map[string]interface{}{
		"apiVersion": "grove.io/v1alpha1",
		"kind":       "PodCliqueSet",
		"metadata": map[string]interface{}{
			"name":      "graph",
			"namespace": "default",
		},
		"spec": map[string]interface{}{
			"replicas": int64(1),
			"template": map[string]interface{}{
				"cliques": []interface{}{
					map[string]interface{}{
						"name": "frontend",
						"spec": map[string]interface{}{
							"roleName":     "frontend",
							"replicas":     int64(1),
							"minAvailable": int64(1),
							"podSpec": map[string]interface{}{
								"containers": []interface{}{},
							},
						},
						"topologyConstraint": map[string]interface{}{
							"futureProviderField": "preserved",
						},
					},
				},
			},
		},
	}}

	t.Log("Record dry-run, apply, and checkpoint operations")
	var applyDryRuns []bool
	providerProgramReads := 0
	providerFieldSent := false
	kubeClient := fake.NewClientBuilder().
		WithScheme(scheme).
		WithObjects(dgd).
		WithInterceptorFuncs(interceptor.Funcs{
			Get: func(
				ctx context.Context,
				reader client.WithWatch,
				key client.ObjectKey,
				obj client.Object,
				opts ...client.GetOption,
			) error {
				if _, ok := obj.(*grovev1alpha1.PodCliqueSet); ok {
					providerProgramReads++
				}
				return reader.Get(ctx, key, obj, opts...)
			},
			Apply: func(
				ctx context.Context,
				writer client.WithWatch,
				obj runtime.ApplyConfiguration,
				opts ...client.ApplyOption,
			) error {
				options := (&client.ApplyOptions{}).ApplyOptions(opts)
				applyDryRuns = append(applyDryRuns, len(options.DryRun) != 0)

				// Inspect the future-aware payload without relying on unchecked shapes.
				payload, ok := obj.(interface{ UnstructuredContent() map[string]interface{} })
				require.True(t, ok, "apply configuration must expose unstructured content")
				cliques, found, nestedErr := unstructured.NestedSlice(payload.UnstructuredContent(), "spec", "template", "cliques")
				require.NoError(t, nestedErr)
				require.True(t, found)
				require.NotEmpty(t, cliques)
				clique, ok := cliques[0].(map[string]interface{})
				require.True(t, ok, "first clique must be an object")
				constraint, ok := clique["topologyConstraint"].(map[string]interface{})
				require.True(t, ok, "topologyConstraint must be an object")
				providerFieldSent = providerFieldSent || constraint["futureProviderField"] == "preserved"
				return writer.Apply(ctx, obj, opts...)
			},
		}).
		Build()
	recorder := events.NewFakeRecorder(10)
	reconciler := &groveWorkloadsReconciler{
		syncer: newDGDResourceSyncer(kubeClient, recorder),
	}

	t.Log("Reconcile the provider program through strict dry-run and SSA")
	result, err := reconciler.reconcileGroveProviderProgram(context.Background(), dgd, desired)

	t.Log("Verify validation preceded mutation without reading the successful write back")
	require.NoError(t, err)
	assert.Equal(t, []bool{true, false}, applyDryRuns)
	assert.Equal(t, 1, providerProgramReads)
	assert.Equal(t, int32(1), result.Spec.Replicas)
	select {
	case event := <-recorder.Events:
		assert.Contains(t, event, "CreatePodCliqueSet")
	default:
		t.Fatal("provider workload mutation event was not emitted")
	}

	t.Log("Verify the SSA payload retained the unknown provider field")
	assert.True(t, providerFieldSent, "unstructured provider field must be present in the SSA payload")

	t.Log("Verify the applied workload remains readable through the typed Grove API")
	live := &grovev1alpha1.PodCliqueSet{}
	require.NoError(t, kubeClient.Get(context.Background(), client.ObjectKey{Name: "graph", Namespace: "default"}, live))
	assert.Equal(t, int32(1), live.Spec.Replicas)
}

func TestGroveProviderOverrideReconcileAppliesRenderedMetadataChanges(t *testing.T) {
	t.Log("Build one override-backed program and record provider API calls")
	scheme := newDynamoGraphDeploymentControllerTestScheme(t)
	dgd := &nvidiacomv1beta1.DynamoGraphDeployment{
		TypeMeta: metav1.TypeMeta{APIVersion: nvidiacomv1beta1.GroupVersion.String(), Kind: "DynamoGraphDeployment"},
		ObjectMeta: metav1.ObjectMeta{
			Name:      "graph",
			Namespace: "default",
			UID:       types.UID("graph-uid"),
		},
	}
	mutatingApplies := 0
	dryRuns := 0
	rejectDryRun := false
	kubeClient := fake.NewClientBuilder().
		WithScheme(scheme).
		WithObjects(dgd).
		WithInterceptorFuncs(interceptor.Funcs{
			Apply: func(
				ctx context.Context,
				writer client.WithWatch,
				obj runtime.ApplyConfiguration,
				opts ...client.ApplyOption,
			) error {
				options := (&client.ApplyOptions{}).ApplyOptions(opts)
				if len(options.DryRun) != 0 {
					dryRuns++
					if rejectDryRun {
						return apierrors.NewServiceUnavailable("provider webhook unavailable")
					}
				} else {
					mutatingApplies++
				}
				return writer.Apply(ctx, obj, opts...)
			},
		}).
		Build()
	reconciler := &groveWorkloadsReconciler{
		syncer: newDGDResourceSyncer(kubeClient, events.NewFakeRecorder(10)),
	}

	t.Log("Apply the initial rendered Volcano queue annotation")
	initial := providerProgramWithReplicas(1)
	initial.SetLabels(map[string]string{"app.kubernetes.io/managed-by": "dynamo"})
	initial.SetAnnotations(map[string]string{consts.GroveAnnotationVolcanoQueue: "queue-a"})
	_, err := reconciler.reconcileGroveProviderProgram(context.Background(), dgd, initial)
	require.NoError(t, err)

	t.Log("Change only rendered metadata and verify the PCS is applied again")
	mutatingApplies = 0
	dryRuns = 0
	updated := providerProgramWithReplicas(1)
	updated.SetLabels(map[string]string{"app.kubernetes.io/managed-by": "dynamo"})
	updated.SetAnnotations(map[string]string{consts.GroveAnnotationVolcanoQueue: "queue-b"})
	_, err = reconciler.reconcileGroveProviderProgram(context.Background(), dgd, updated)
	require.NoError(t, err)
	assert.Equal(t, 1, mutatingApplies)
	assert.Equal(t, 1, dryRuns)
	live := &grovev1alpha1.PodCliqueSet{}
	require.NoError(t, kubeClient.Get(context.Background(), client.ObjectKey{Name: "graph", Namespace: "default"}, live))
	assert.Equal(t, "queue-b", live.Annotations[consts.GroveAnnotationVolcanoQueue])

	t.Log("Keep an unchanged healthy workload ready while provider validation is unavailable")
	mutatingApplies = 0
	dryRuns = 0
	rejectDryRun = true
	unchanged := providerProgramWithReplicas(1)
	unchanged.SetLabels(map[string]string{"app.kubernetes.io/managed-by": "dynamo"})
	unchanged.SetAnnotations(map[string]string{consts.GroveAnnotationVolcanoQueue: "queue-b"})
	_, err = reconciler.reconcileGroveProviderProgram(context.Background(), dgd, unchanged)
	require.NoError(t, err)
	assert.Zero(t, mutatingApplies)
	assert.Zero(t, dryRuns)

	t.Log("Restore drift in an operator-owned label")
	rejectDryRun = false
	live.Labels["app.kubernetes.io/managed-by"] = "external"
	require.NoError(t, kubeClient.Update(context.Background(), live))
	mutatingApplies = 0
	dryRuns = 0
	desired := providerProgramWithReplicas(1)
	desired.SetLabels(map[string]string{"app.kubernetes.io/managed-by": "dynamo"})
	desired.SetAnnotations(map[string]string{consts.GroveAnnotationVolcanoQueue: "queue-b"})
	_, err = reconciler.reconcileGroveProviderProgram(context.Background(), dgd, desired)
	require.NoError(t, err)
	assert.Equal(t, 1, mutatingApplies)
	assert.Equal(t, 1, dryRuns)
	require.NoError(t, kubeClient.Get(context.Background(), client.ObjectKey{Name: "graph", Namespace: "default"}, live))
	assert.Equal(t, "dynamo", live.Labels["app.kubernetes.io/managed-by"])

	t.Log("Restore drift in an operator-owned annotation")
	live.Annotations[consts.GroveAnnotationVolcanoQueue] = "external"
	require.NoError(t, kubeClient.Update(context.Background(), live))
	mutatingApplies = 0
	dryRuns = 0
	desired = providerProgramWithReplicas(1)
	desired.SetLabels(map[string]string{"app.kubernetes.io/managed-by": "dynamo"})
	desired.SetAnnotations(map[string]string{consts.GroveAnnotationVolcanoQueue: "queue-b"})
	_, err = reconciler.reconcileGroveProviderProgram(context.Background(), dgd, desired)
	require.NoError(t, err)
	assert.Equal(t, 1, mutatingApplies)
	assert.Equal(t, 1, dryRuns)
	require.NoError(t, kubeClient.Get(context.Background(), client.ObjectKey{Name: "graph", Namespace: "default"}, live))
	assert.Equal(t, "queue-b", live.Annotations[consts.GroveAnnotationVolcanoQueue])

	t.Log("Restore drift in the DGD controller reference")
	controllerRef := metav1.GetControllerOfNoCopy(live)
	require.NotNil(t, controllerRef)
	controllerRef.Name = "other-graph"
	require.NoError(t, kubeClient.Update(context.Background(), live))
	mutatingApplies = 0
	dryRuns = 0
	desired = providerProgramWithReplicas(1)
	desired.SetLabels(map[string]string{"app.kubernetes.io/managed-by": "dynamo"})
	desired.SetAnnotations(map[string]string{consts.GroveAnnotationVolcanoQueue: "queue-b"})
	_, err = reconciler.reconcileGroveProviderProgram(context.Background(), dgd, desired)
	require.NoError(t, err)
	assert.Equal(t, 1, mutatingApplies)
	assert.Equal(t, 1, dryRuns)
	require.NoError(t, kubeClient.Get(context.Background(), client.ObjectKey{Name: "graph", Namespace: "default"}, live))
	controllerRef = metav1.GetControllerOfNoCopy(live)
	require.NotNil(t, controllerRef)
	assert.Equal(t, dgd.Name, controllerRef.Name)
}

func TestGroveProviderOverrideReconcilePreservesLiveProgramWhenApplyFails(t *testing.T) {
	t.Log("Build a live provider program and a different desired override")
	scheme := newDynamoGraphDeploymentControllerTestScheme(t)
	dgd := &nvidiacomv1beta1.DynamoGraphDeployment{
		TypeMeta: metav1.TypeMeta{APIVersion: nvidiacomv1beta1.GroupVersion.String(), Kind: "DynamoGraphDeployment"},
		ObjectMeta: metav1.ObjectMeta{
			Name:       "graph",
			Namespace:  "default",
			UID:        types.UID("graph-uid"),
			Generation: 2,
		},
		Spec: nvidiacomv1beta1.DynamoGraphDeploymentSpec{
			ProviderOverride: &nvidiacomv1beta1.ProviderOverride{
				APIVersion: provideroverride.GroveAPIVersion,
				Target:     provideroverride.TargetPodCliqueSet,
				Value:      apiextensionsv1.JSON{Raw: []byte(`{"spec":{"replicas":2}}`)},
			},
		},
	}
	existing := &grovev1alpha1.PodCliqueSet{
		ObjectMeta: metav1.ObjectMeta{Name: "graph", Namespace: "default"},
		Spec:       grovev1alpha1.PodCliqueSetSpec{Replicas: 1},
	}

	t.Log("Allow strict dry-run but reject the mutating SSA write")
	kubeClient := fake.NewClientBuilder().
		WithScheme(scheme).
		WithObjects(dgd, existing).
		WithInterceptorFuncs(interceptor.Funcs{
			Apply: func(
				ctx context.Context,
				writer client.WithWatch,
				obj runtime.ApplyConfiguration,
				opts ...client.ApplyOption,
			) error {
				options := (&client.ApplyOptions{}).ApplyOptions(opts)
				if len(options.DryRun) != 0 {
					return nil
				}
				return apierrors.NewServiceUnavailable("injected provider-program apply failure")
			},
		}).
		Build()
	reconciler := &groveWorkloadsReconciler{
		syncer: newDGDResourceSyncer(kubeClient, events.NewFakeRecorder(10)),
	}

	t.Log("Reconcile and stop after the failed mutating apply")
	_, err := reconciler.reconcileGroveProviderProgram(context.Background(), dgd, providerProgramWithReplicas(2))
	require.ErrorContains(t, err, "apply PodCliqueSet provider program")

	t.Log("Verify the previously live PCS remains unchanged")
	live := &grovev1alpha1.PodCliqueSet{}
	require.NoError(t, kubeClient.Get(context.Background(), client.ObjectKey{Name: "graph", Namespace: "default"}, live))
	assert.Equal(t, int32(1), live.Spec.Replicas)
}

func TestGroveProviderOverrideReconcileReturnsAppliedProgramWhenMarkerPatchFails(t *testing.T) {
	t.Log("Build one desired override-backed program")
	scheme := newDynamoGraphDeploymentControllerTestScheme(t)
	dgd := &nvidiacomv1beta1.DynamoGraphDeployment{
		TypeMeta: metav1.TypeMeta{APIVersion: nvidiacomv1beta1.GroupVersion.String(), Kind: "DynamoGraphDeployment"},
		ObjectMeta: metav1.ObjectMeta{
			Name:      "graph",
			Namespace: "default",
			UID:       types.UID("graph-uid"),
		},
		Spec: nvidiacomv1beta1.DynamoGraphDeploymentSpec{
			ProviderOverride: &nvidiacomv1beta1.ProviderOverride{
				APIVersion: provideroverride.GroveAPIVersion,
				Target:     provideroverride.TargetPodCliqueSet,
				Value:      apiextensionsv1.JSON{Raw: []byte(`{"spec":{"replicas":2}}`)},
			},
		},
	}

	t.Log("Allow SSA but fail the bookkeeping marker patch")
	kubeClient := fake.NewClientBuilder().
		WithScheme(scheme).
		WithObjects(dgd).
		WithInterceptorFuncs(interceptor.Funcs{
			Patch: func(
				context.Context,
				client.WithWatch,
				client.Object,
				client.Patch,
				...client.PatchOption,
			) error {
				return errors.New("injected marker patch failure")
			},
		}).
		Build()
	reconciler := &groveWorkloadsReconciler{
		syncer: newDGDResourceSyncer(kubeClient, events.NewFakeRecorder(10)),
	}

	t.Log("Reconcile and return the PCS already applied by SSA")
	result, err := reconciler.reconcileGroveProviderProgram(context.Background(), dgd, providerProgramWithReplicas(2))
	require.ErrorContains(t, err, "record applied PodCliqueSet provider program")
	require.NotNil(t, result)
	assert.Equal(t, int32(2), result.Spec.Replicas)

	t.Log("Verify the applied PCS remains live even though its marker was not recorded")
	live := &grovev1alpha1.PodCliqueSet{}
	require.NoError(t, kubeClient.Get(context.Background(), client.ObjectKey{Name: "graph", Namespace: "default"}, live))
	assert.Equal(t, int32(2), live.Spec.Replicas)
	assert.NotContains(t, live.Annotations, commoncontroller.NvidiaAnnotationGenerationKey)
}

func TestGroveProviderOverrideReconcileDoesNotMutateWhenDryRunFails(t *testing.T) {
	t.Log("Build an existing workload and a materially different desired program")
	scheme := newDynamoGraphDeploymentControllerTestScheme(t)
	dgd := &nvidiacomv1beta1.DynamoGraphDeployment{
		TypeMeta: metav1.TypeMeta{APIVersion: nvidiacomv1beta1.GroupVersion.String(), Kind: "DynamoGraphDeployment"},
		ObjectMeta: metav1.ObjectMeta{
			Name:      "graph",
			Namespace: "default",
			UID:       types.UID("graph-uid"),
		},
	}
	existing := &grovev1alpha1.PodCliqueSet{
		ObjectMeta: metav1.ObjectMeta{Name: "graph", Namespace: "default"},
		Spec:       grovev1alpha1.PodCliqueSetSpec{Replicas: 1},
	}
	desired := &unstructured.Unstructured{Object: map[string]interface{}{
		"apiVersion": "grove.io/v1alpha1",
		"kind":       "PodCliqueSet",
		"metadata": map[string]interface{}{
			"name":      "graph",
			"namespace": "default",
		},
		"spec": map[string]interface{}{
			"replicas": int64(7),
			"template": map[string]interface{}{"cliques": []interface{}{}},
		},
	}}
	t.Log("Fail the API-server dry-run and record any real mutation attempt")
	actualApplies := 0
	kubeClient := fake.NewClientBuilder().
		WithScheme(scheme).
		WithObjects(dgd, existing).
		WithInterceptorFuncs(interceptor.Funcs{
			Apply: func(
				ctx context.Context,
				writer client.WithWatch,
				obj runtime.ApplyConfiguration,
				opts ...client.ApplyOption,
			) error {
				options := (&client.ApplyOptions{}).ApplyOptions(opts)
				if len(options.DryRun) != 0 {
					return apierrors.NewServiceUnavailable("provider webhook unavailable")
				}
				actualApplies++
				return writer.Apply(ctx, obj, opts...)
			},
		}).
		Build()
	reconciler := &groveWorkloadsReconciler{
		syncer: newDGDResourceSyncer(kubeClient, events.NewFakeRecorder(10)),
	}

	t.Log("Attempt to reconcile the invalid provider program")
	_, err := reconciler.reconcileGroveProviderProgram(context.Background(), dgd, desired)

	t.Log("Verify reconciliation stopped before a mutating apply")
	require.ErrorContains(t, err, "validate Grove provider program")
	assert.Zero(t, actualApplies)
	reason, classified := workloadProgramFailureReason(err)
	require.True(t, classified)
	assert.Equal(t, reasonProviderValidationUnavailable, reason)

	t.Log("Verify the previously applied workload was preserved")
	live := &grovev1alpha1.PodCliqueSet{}
	require.NoError(t, kubeClient.Get(context.Background(), client.ObjectKey{Name: "graph", Namespace: "default"}, live))
	assert.Equal(t, int32(1), live.Spec.Replicas)
}

func TestGroveProviderPreflightFailureReason(t *testing.T) {
	tests := []struct {
		name       string
		err        error
		liveExists bool
		want       Reason
	}{
		{
			name:       "explicit provider rejection is invalid",
			err:        apierrors.NewBadRequest("provider rejected desired program"),
			liveExists: true,
			want:       reasonProviderOverrideInvalid,
		},
		{
			name: "Grove webhook admission denial is invalid",
			err: &apierrors.StatusError{ErrStatus: metav1.Status{
				Status:  metav1.StatusFailure,
				Code:    http.StatusForbidden,
				Reason:  metav1.StatusReasonForbidden,
				Message: "spec.template: Invalid value",
			}},
			liveExists: true,
			want:       reasonProviderOverrideInvalid,
		},
		{
			name:       "unclassified API failure is unavailable",
			err:        errors.New("connection reset"),
			liveExists: true,
			want:       reasonProviderValidationUnavailable,
		},
		{
			name:       "internal API failure is unavailable",
			err:        apierrors.NewInternalError(errors.New("provider failed")),
			liveExists: true,
			want:       reasonProviderValidationUnavailable,
		},
		{
			name: "authorization failure is unavailable",
			err: apierrors.NewForbidden(
				schema.GroupResource{Group: "grove.io", Resource: "podcliquesets"},
				"graph",
				errors.New("denied"),
			),
			liveExists: true,
			want:       reasonProviderValidationUnavailable,
		},
		{
			name: "absent backing program remains unavailable",
			err:  errors.New("connection reset"),
			want: reasonProviderProgramUnavailable,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Log("Classify the API-server preflight failure")
			assert.Equal(t, tt.want, groveProviderPreflightFailureReason(tt.err, tt.liveExists))
		})
	}
}

func TestGroveProviderOverridePreservesLiveProgramAndReportsUnavailable(t *testing.T) {
	t.Log("Build one override-backed provider program")
	scheme := newDynamoGraphDeploymentControllerTestScheme(t)
	dgd := &nvidiacomv1beta1.DynamoGraphDeployment{
		TypeMeta: metav1.TypeMeta{APIVersion: nvidiacomv1beta1.GroupVersion.String(), Kind: "DynamoGraphDeployment"},
		ObjectMeta: metav1.ObjectMeta{
			Name:      "graph",
			Namespace: "default",
			UID:       types.UID("graph-uid"),
		},
		Spec: nvidiacomv1beta1.DynamoGraphDeploymentSpec{
			ProviderOverride: &nvidiacomv1beta1.ProviderOverride{
				APIVersion: provideroverride.GroveAPIVersion,
				Target:     provideroverride.TargetPodCliqueSet,
				Value: apiextensionsv1.JSON{Raw: []byte(
					`{"spec":{"template":{"topologyConstraint":{"pack":{"required":"rack"}}}}}`,
				)},
			},
		},
	}
	desired := providerProgramWithReplicas(1)
	mutatingApplies := 0
	kubeClient := fake.NewClientBuilder().
		WithScheme(scheme).
		WithObjects(dgd).
		WithInterceptorFuncs(interceptor.Funcs{
			Apply: func(
				ctx context.Context,
				writer client.WithWatch,
				obj runtime.ApplyConfiguration,
				opts ...client.ApplyOption,
			) error {
				options := (&client.ApplyOptions{}).ApplyOptions(opts)
				payload := obj.(interface{ UnstructuredContent() map[string]interface{} })
				replicas, _, nestedErr := unstructured.NestedInt64(payload.UnstructuredContent(), "spec", "replicas")
				require.NoError(t, nestedErr)
				if len(options.DryRun) != 0 && replicas == 7 {
					return apierrors.NewBadRequest("provider rejected desired program")
				}
				if len(options.DryRun) == 0 {
					mutatingApplies++
				}
				return writer.Apply(ctx, obj, opts...)
			},
		}).
		Build()
	reconciler := &groveWorkloadsReconciler{
		syncer: newDGDResourceSyncer(kubeClient, events.NewFakeRecorder(10)),
	}
	applied, err := reconciler.reconcileGroveProviderProgram(context.Background(), dgd, desired)
	require.NoError(t, err)
	require.NotNil(t, applied)
	assert.Equal(t, int32(1), applied.Spec.Replicas)

	t.Log("Apply a valid override update directly to the existing PCS")
	dgd.Generation = 2
	dgd.Spec.ProviderOverride.Value.Raw = []byte(
		`{"spec":{"template":{"topologyConstraint":{"pack":{"required":"host"}}}}}`,
	)
	mutatingApplies = 0
	updated, err := reconciler.reconcileGroveProviderProgram(context.Background(), dgd, providerProgramWithReplicas(2))
	require.NoError(t, err)
	assert.Equal(t, 1, mutatingApplies)
	require.NotNil(t, updated)
	assert.Equal(t, int32(2), updated.Spec.Replicas)

	t.Log("Propose a provider-rejected update while the updated PCS remains live")
	dgd.Generation = 3
	dgd.Spec.ProviderOverride.Value.Raw = []byte(
		`{"spec":{"template":{"topologyConstraint":{"pack":{"required":"rack"}}}}}`,
	)
	mutatingApplies = 0
	_, err = reconciler.reconcileGroveProviderProgram(context.Background(), dgd, providerProgramWithReplicas(7))
	require.ErrorContains(t, err, "validate Grove provider program")
	assert.Zero(t, mutatingApplies)
	reason, classified := workloadProgramFailureReason(err)
	require.True(t, classified)
	assert.Equal(t, reasonProviderOverrideInvalid, reason)

	t.Log("Verify rejected desired state left the applied PCS untouched")
	live := &grovev1alpha1.PodCliqueSet{}
	require.NoError(t, kubeClient.Get(context.Background(), client.ObjectKey{Namespace: "default", Name: "graph"}, live))
	assert.Equal(t, int32(2), live.Spec.Replicas)

	t.Log("Persist an unsupported provider envelope while the applied PCS remains live")
	dgd.Generation = 4
	dgd.Spec.ProviderOverride.APIVersion = "grove.io/v2"
	mutatingApplies = 0
	rendered := &grovev1alpha1.PodCliqueSet{
		ObjectMeta: metav1.ObjectMeta{Name: "graph", Namespace: "default"},
		Spec: grovev1alpha1.PodCliqueSetSpec{
			Replicas: 7,
			Template: grovev1alpha1.PodCliqueSetTemplateSpec{
				Cliques: []*grovev1alpha1.PodCliqueTemplateSpec{},
			},
		},
	}
	_, err = reconciler.reconcileRenderedPodCliqueSet(context.Background(), dgd, rendered)
	require.ErrorContains(t, err, "unsupported Grove apiVersion")
	assert.Zero(t, mutatingApplies)
	reason, classified = workloadProgramFailureReason(err)
	require.True(t, classified)
	assert.Equal(t, reasonProviderOverrideInvalid, reason)

	t.Log("Delete the live PCS while desired state remains unsupported")
	require.NoError(t, kubeClient.Delete(context.Background(), live))
	mutatingApplies = 0
	_, err = reconciler.reconcileRenderedPodCliqueSet(context.Background(), dgd, rendered)
	require.ErrorContains(t, err, "unsupported Grove apiVersion")
	assert.Zero(t, mutatingApplies)
	reason, classified = workloadProgramFailureReason(err)
	require.True(t, classified)
	assert.Equal(t, reasonProviderProgramUnavailable, reason)

	t.Log("Correct the desired provider identity and recreate the backing PCS")
	dgd.Generation = 5
	dgd.Spec.ProviderOverride.APIVersion = provideroverride.GroveAPIVersion
	mutatingApplies = 0
	recreated, err := reconciler.reconcileRenderedPodCliqueSet(context.Background(), dgd, &grovev1alpha1.PodCliqueSet{
		ObjectMeta: metav1.ObjectMeta{Name: "graph", Namespace: "default"},
		Spec: grovev1alpha1.PodCliqueSetSpec{
			Replicas: 3,
			Template: grovev1alpha1.PodCliqueSetTemplateSpec{
				Cliques: []*grovev1alpha1.PodCliqueTemplateSpec{},
			},
		},
	})
	require.NoError(t, err)
	require.NotNil(t, recreated)
	assert.Equal(t, int32(3), recreated.Spec.Replicas)
	assert.Equal(t, 1, mutatingApplies)
}

func providerProgramWithReplicas(replicas int64) *unstructured.Unstructured {
	return &unstructured.Unstructured{Object: map[string]interface{}{
		"apiVersion": provideroverride.GroveAPIVersion,
		"kind":       provideroverride.TargetPodCliqueSet,
		"metadata": map[string]interface{}{
			"name":      "graph",
			"namespace": "default",
		},
		"spec": map[string]interface{}{
			"replicas": replicas,
			"template": map[string]interface{}{"cliques": []interface{}{}},
		},
	}}
}
