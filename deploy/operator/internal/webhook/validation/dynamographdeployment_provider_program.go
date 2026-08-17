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

package validation

import (
	"context"
	"errors"
	"fmt"
	"reflect"
	"strings"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	commoncontroller "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/provideroverride"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

// groveProviderProgramValidator renders and asks the installed Grove API to
// validate provider programs before DGD admission accepts them.
type groveProviderProgramValidator struct {
	client           client.Client
	config           *configv1alpha1.OperatorConfiguration
	runtimeConfig    *commoncontroller.RuntimeConfig
	secretsRetriever dynamo.SecretsRetriever
}

// ValidateCreate validates a provider program only when the proposed DGD uses
// a Grove override. dgd must not be nil.
func (v *groveProviderProgramValidator) ValidateCreate(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
) error {
	if !requiresGroveProviderProgramValidation(dgd) {
		return nil
	}
	return v.validate(ctx, dgd)
}

// ValidateUpdate validates only updates that can change an override-backed
// Grove provider program. oldDGD and newDGD must not be nil.
func (v *groveProviderProgramValidator) ValidateUpdate(
	ctx context.Context,
	oldDGD, newDGD *nvidiacomv1beta1.DynamoGraphDeployment,
) error {
	// Avoid rendering only when neither typed fields nor render-relevant metadata changed.
	if !groveProviderProgramInputsChanged(oldDGD, newDGD) {
		return nil
	}
	if !requiresGroveProviderProgramValidation(oldDGD) && !requiresGroveProviderProgramValidation(newDGD) {
		return nil
	}
	return v.validate(ctx, newDGD)
}

// validate renders and dry-runs the complete proposed workload. dgd must not
// be nil.
func (v *groveProviderProgramValidator) validate(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
) error {
	// Render the same provider resource kind used by Grove reconciliation.
	desired, err := dynamo.GenerateGrovePodCliqueSet(
		ctx,
		dgd,
		v.config,
		v.runtimeConfig,
		v.client,
		v.secretsRetriever,
		nil,
		nil,
		nil,
	)
	if err != nil {
		return invalidProviderProgram(dgd, fmt.Errorf("render Grove PodCliqueSet: %w", err))
	}

	// Insert opaque provider-owned subtrees before server-side schema validation.
	providerObject, err := provideroverride.ApplyGroveOverrides(dgd, desired)
	if err != nil {
		return invalidProviderProgram(dgd, fmt.Errorf("apply Grove provider overrides: %w", err))
	}

	// Fail closed when the installed provider API or its admission chain rejects the program.
	if err := provideroverride.DryRunGroveProgram(ctx, v.client, providerObject); err != nil {
		return invalidProviderProgram(dgd, fmt.Errorf("dry-run Grove PodCliqueSet: %s", providerProgramValidationDetail(err)))
	}
	return nil
}

func providerProgramValidationDetail(err error) string {
	// Prefer structured API causes because client wrappers include request-specific object identities.
	var status apierrors.APIStatus
	if !errors.As(err, &status) || status.Status().Details == nil {
		detail := err.Error()
		if index := strings.Index(detail, "): ."); index >= 0 {
			return detail[index+3:]
		}
		return detail
	}

	// Join all provider admission causes without exposing the submitted override value.
	causes := status.Status().Details.Causes
	details := make([]string, 0, len(causes))
	for _, cause := range causes {
		details = append(details, strings.TrimSpace(cause.Field+": "+cause.Message))
	}
	if len(details) == 0 {
		return err.Error()
	}
	return strings.Join(details, "; ")
}

// groveProviderProgramInputsChanged reports whether a DGD update can alter the
// rendered Grove program. oldDGD and newDGD must not be nil.
func groveProviderProgramInputsChanged(
	oldDGD, newDGD *nvidiacomv1beta1.DynamoGraphDeployment,
) bool {
	return !reflect.DeepEqual(oldDGD.Spec, newDGD.Spec) ||
		!reflect.DeepEqual(
			dynamo.GroveProgramAnnotations(oldDGD.Annotations),
			dynamo.GroveProgramAnnotations(newDGD.Annotations),
		)
}

func requiresGroveProviderProgramValidation(dgd *nvidiacomv1beta1.DynamoGraphDeployment) bool {
	// Provider dry-run is relevant only after sticky Grove selection and with an override lifecycle.
	return dgd.Annotations[consts.KubeAnnotationWorkloadProvider] == consts.WorkloadProviderGrove &&
		provideroverride.HasGroveOverrides(dgd)
}

func invalidProviderProgram(dgd *nvidiacomv1beta1.DynamoGraphDeployment, err error) error {
	return apierrors.NewInvalid(
		schema.GroupKind{Group: nvidiacomv1beta1.GroupVersion.Group, Kind: "DynamoGraphDeployment"},
		dgd.Name,
		field.ErrorList{field.Invalid(field.NewPath("spec"), "provider program", err.Error())},
	)
}
