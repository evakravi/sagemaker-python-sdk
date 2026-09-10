# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
"""Telemetry emitted by a real JumpStart ModelBuilder flow.

The JumpStart metadata, the image URI, and the model artifact location come from the live
JumpStart catalog. SageMaker resource creation and the telemetry GET requests are intercepted,
so the test creates no endpoint and sends no telemetry.
"""
from __future__ import absolute_import

from unittest.mock import Mock, patch
from urllib.parse import parse_qs, urlsplit

import requests

from sagemaker.session import Session
from sagemaker.serve.builder.model_builder import ModelBuilder
from sagemaker.serve.builder.schema_builder import SchemaBuilder
from sagemaker.user_agent import SDK_VERSION

ROLE_NAME = "SageMakerRole"
JS_MMS_MODEL_ID = "huggingface-sentencesimilarity-bge-m3"
SAMPLE_MMS_PROMPT = ["How cute your dog is!"]
SAMPLE_MMS_RESPONSE = {"embedding": []}
LEGACY_BUCKET_PREFIX = "dev-exp-t-"
SDK_BUCKET_PREFIX = "sm-pysdk-t-"
JUMPSTART_FEATURE_CODE = "8"


def _telemetry_query(url):
    return {key: values[0] for key, values in parse_qs(urlsplit(url).query).items()}


def _events(telemetry_requests, bucket_prefix, func_name):
    return [
        _telemetry_query(url)
        for url in telemetry_requests
        if urlsplit(url).netloc.startswith(bucket_prefix)
        and _telemetry_query(url).get("x-extra") == func_name
    ]


def test_jumpstart_model_builder_emits_legacy_and_sdk_telemetry(sagemaker_session, account):
    telemetry_requests = []

    def record_telemetry_request(url, *args, **kwargs):
        if "/telemetry?" in url:
            telemetry_requests.append(url)
        return Mock(status_code=200)

    with (
        patch.object(requests, "get", side_effect=record_telemetry_request),
        patch.object(Session, "create_model", return_value="mock_model"),
        patch.object(Session, "endpoint_from_production_variants", return_value="mock_endpoint"),
    ):
        model_builder = ModelBuilder(
            model=JS_MMS_MODEL_ID,
            schema_builder=SchemaBuilder(SAMPLE_MMS_PROMPT, SAMPLE_MMS_RESPONSE),
            role_arn=f"arn:aws:iam::{account}:role/{ROLE_NAME}",
            sagemaker_session=sagemaker_session,
            instance_type="ml.m5.xlarge",
        )
        model = model_builder.build()
        model.deploy(instance_type="ml.m5.xlarge", endpoint_logging=False)

    legacy_build_events = _events(telemetry_requests, LEGACY_BUCKET_PREFIX, "ModelBuilder.build")
    assert len(legacy_build_events) == 1
    legacy_build_event = legacy_build_events[0]
    assert legacy_build_event["x-accountId"] == account
    assert legacy_build_event["x-mode"] == "3"
    assert legacy_build_event["x-modelHub"] == "1"
    assert legacy_build_event["x-sdkVersion"] == SDK_VERSION
    assert "x-feature" not in legacy_build_event
    assert "x-jumpstartModelId" not in legacy_build_event

    sdk_build_events = _events(telemetry_requests, SDK_BUCKET_PREFIX, "ModelBuilder.build")
    assert len(sdk_build_events) == 1
    sdk_build_event = sdk_build_events[0]
    assert sdk_build_event["x-accountId"] == account
    assert sdk_build_event["x-feature"] == JUMPSTART_FEATURE_CODE
    assert sdk_build_event["x-jumpstartModelId"] == JS_MMS_MODEL_ID
    assert sdk_build_event["x-mode"] == "SAGEMAKER_ENDPOINT"
    assert sdk_build_event["x-sdkVersion"] == SDK_VERSION
    assert {"x-env", "x-sys", "x-platform", "x-latency"} <= set(sdk_build_event)
    assert "x-modelHub" not in sdk_build_event

    sdk_deploy_events = _events(telemetry_requests, SDK_BUCKET_PREFIX, "jumpstart_model.deploy")
    assert len(sdk_deploy_events) == 1
    assert sdk_deploy_events[0]["x-feature"] == JUMPSTART_FEATURE_CODE
    assert sdk_deploy_events[0]["x-jumpstartModelId"] == JS_MMS_MODEL_ID

    legacy_deploy_events = _events(telemetry_requests, LEGACY_BUCKET_PREFIX, "jumpstart.deploy")
    assert len(legacy_deploy_events) == 1
    assert legacy_deploy_events[0]["x-modelHub"] == "1"
    assert "x-jumpstartModelId" not in legacy_deploy_events[0]

    sdk_bucket_events = [
        _telemetry_query(url)
        for url in telemetry_requests
        if urlsplit(url).netloc.startswith(SDK_BUCKET_PREFIX)
    ]
    assert all(event["x-feature"] == JUMPSTART_FEATURE_CODE for event in sdk_bucket_events)
    assert all("x-modelHub" not in event for event in sdk_bucket_events)
