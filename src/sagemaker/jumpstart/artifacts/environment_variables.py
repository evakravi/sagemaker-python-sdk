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
"""This module contains functions for obtaining JumpStart environment variables."""
from __future__ import absolute_import
from typing import Dict, Optional
from sagemaker.jumpstart.constants import (
    JUMPSTART_DEFAULT_REGION_NAME,
)
from sagemaker.jumpstart.enums import (
    JumpStartScriptScope,
)
from sagemaker.jumpstart.utils import (
    verify_model_region_and_return_specs,
)


def _retrieve_default_environment_variables(
    model_id: str,
    model_version: str,
    region: Optional[str] = None,
    tolerate_vulnerable_model: bool = False,
    tolerate_deprecated_model: bool = False,
    include_aws_sdk_env_vars: bool = True,
) -> Dict[str, str]:
    """Retrieves the inference environment variables for the model matching the given arguments.

    Args:
        model_id (str): JumpStart model ID of the JumpStart model for which to
            retrieve the default environment variables.
        model_version (str): Version of the JumpStart model for which to retrieve the
            default environment variables.
        region (Optional[str]): Region for which to retrieve default environment variables.
            (Default: None).
        tolerate_vulnerable_model (bool): True if vulnerable versions of model
            specifications should be tolerated (exception not raised). If False, raises an
            exception if the script used by this version of the model has dependencies with known
            security vulnerabilities. (Default: False).
        tolerate_deprecated_model (bool): True if deprecated versions of model
            specifications should be tolerated (exception not raised). If False, raises
            an exception if the version of the model is deprecated. (Default: False).
        include_aws_sdk_env_vars (bool): True if environment variables for low-level AWS API call
            should be included. The `Model` class of the SageMaker Python SDK inserts environment
            variables that would be required when making the low-level AWS API call.
            (Default: True).

    Returns:
        dict: the inference environment variables to use for the model.
    """

    if region is None:
        region = JUMPSTART_DEFAULT_REGION_NAME

    model_specs = verify_model_region_and_return_specs(
        model_id=model_id,
        version=model_version,
        scope=JumpStartScriptScope.INFERENCE,
        region=region,
        tolerate_vulnerable_model=tolerate_vulnerable_model,
        tolerate_deprecated_model=tolerate_deprecated_model,
    )

    default_environment_variables: Dict[str, str] = {}
    for environment_variable in model_specs.inference_environment_variables:
        if include_aws_sdk_env_vars or (
            not include_aws_sdk_env_vars and environment_variable.required_for_model_class is True
        ):
            default_environment_variables[environment_variable.name] = str(
                environment_variable.default
            )
    return default_environment_variables