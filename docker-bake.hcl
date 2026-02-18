variable "REPOSITORY" {
  default = "quay.io/vllm/automation-vllm-omni"
}

variable "RELEASE_IMAGE" {
  default = false
}

# GITHUB_* variables are set as env vars in github actions
variable "GITHUB_SHA" {}
variable "GITHUB_REPOSITORY" {}
variable "GITHUB_RUN_ID" {}

variable "VLLM_OMNI_VERSION" {}
variable "VLLM_VERSION" {}

variable "PYTHON_VERSION" {
  default = "3.12"
}

variable "ROCM_VERSION" {
  # This can be overridden by the prepare-payload action
  default = "6.4.3"
}


target "docker-metadata-action" {} // populated by gha docker/metadata-action

target "_common" {
  context = "."

  args = {
    BASE_UBI_IMAGE_TAG = "9.6-1760515502"
    PYTHON_VERSION = "3.12"
  }

  inherits = ["docker-metadata-action"]

  platforms = [
    "linux/amd64",
  ]
  labels = {
    "org.opencontainers.image.source" = "https://github.com/${GITHUB_REPOSITORY}"
    "vcs-ref" = "${GITHUB_SHA}"
    "vcs-type" = "git"
    "vllm-version" = "${VLLM_VERSION}"
    "vllm-omni-version" = "${VLLM_OMNI_VERSION}"
  }
}

group "default" {
  targets = [
    "cuda",
    "rocm",
    "cpu",
  ]
}

target "cuda" {
  inherits = ["_common"]
  dockerfile = "Dockerfile.ubi"

  args = {
    PYTHON_VERSION = "${PYTHON_VERSION}"
    CUDA_MAJOR =  "12"
    CUDA_MINOR =  "9"
  }

  tags = [
    "${REPOSITORY}:${replace(VLLM_OMNI_VERSION, "+", "_")}", # version might contain local version specifiers (+) which are not valid tags
    "${REPOSITORY}:cuda-${GITHUB_SHA}",
    "${REPOSITORY}:cuda-${GITHUB_RUN_ID}",
    RELEASE_IMAGE ? "quay.io/vllm/vllm-omni-cuda:${replace(VLLM_OMNI_VERSION, "+", "_")}" : ""
  ]
}

target "rocm" {
  inherits = ["_common"]
  dockerfile = "Dockerfile.rocm.ubi"

  args = {
    PYTHON_VERSION = "${PYTHON_VERSION}"
    ROCM_VERSION = "${ROCM_VERSION}"
  }

  tags = [
    "${REPOSITORY}:${replace(VLLM_OMNI_VERSION, "+", "_")}", # version might contain local version specifiers (+) which are not valid tags
    "${REPOSITORY}:rocm-${GITHUB_SHA}",
    "${REPOSITORY}:rocm-${GITHUB_RUN_ID}",
    RELEASE_IMAGE ? "quay.io/vllm/vllm-omni-rocm:${replace(VLLM_OMNI_VERSION, "+", "_")}" : ""
  ]
}

target "cpu" {
  inherits = ["_common"]
  dockerfile = "Dockerfile.cpu.ubi"

  args = {
    PYTHON_VERSION = "${PYTHON_VERSION}"
  }

  tags = [
    "${REPOSITORY}:${replace(VLLM_OMNI_VERSION, "+", "_")}", # version might contain local version specifiers (+) which are not valid tags
    "${REPOSITORY}:cpu-${GITHUB_SHA}",
    "${REPOSITORY}:cpu-${GITHUB_RUN_ID}",
    RELEASE_IMAGE ? "quay.io/vllm/vllm-omni-cpu:${replace(VLLM_OMNI_VERSION, "+", "_")}" : ""
  ]
}
