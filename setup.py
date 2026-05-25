# SPDX-FileCopyrightText: Copyright 2023-2025 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import glob
import importlib.util
import os
import os.path as osp
import pathlib
import platform
import shutil
import sys

from setuptools import find_packages, setup

__version__ = None
exec(open("gsplat/version.py", "r").read())

URL = "https://github.com/nerfstudio-project/gsplat"

_is_apple_silicon = sys.platform == "darwin" and platform.machine() == "arm64"
_has_cuda_toolkit = bool(os.getenv("CUDA_HOME")) or shutil.which("nvcc") is not None
_default_build_no_cuda = "1" if _is_apple_silicon or not _has_cuda_toolkit else "0"
BUILD_NO_CUDA = os.getenv("BUILD_NO_CUDA", _default_build_no_cuda) == "1"
BUILD_METAL = os.getenv("BUILD_METAL", "1" if _is_apple_silicon else "0") == "1"


def get_ext():
    from torch.utils.cpp_extension import BuildExtension

    return BuildExtension.with_options(no_python_abi_suffix=True, use_ninja=True)


def get_extensions():
    from torch.utils.cpp_extension import CUDAExtension

    # Use the same build parameters as the JIT build. However, directly
    # importing the gsplat.cuda.build module would trigger a circular
    # dependency where gsplat is imported before it is built. To avoid
    # this, we sidestep the traditional Python import mechanism and construct
    # the module directly from build.py.
    spec = importlib.util.spec_from_file_location(
        "gsplat_cuda_build", os.path.join("gsplat", "cuda", "build.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    params = module.get_build_parameters()

    setup_dir = os.path.dirname(os.path.abspath(__file__))
    sources = [os.path.relpath(s, setup_dir) for s in params.sources]

    extension = CUDAExtension(
        "gsplat.csrc",
        sources=sources,
        include_dirs=params.extra_include_paths,
        extra_compile_args={
            "cxx": params.extra_cflags,
            "nvcc": params.extra_cuda_cflags,
        },
        extra_link_args=params.extra_ldflags,
    )
    return [extension]


def get_metal_extensions():
    from torch.utils.cpp_extension import CppExtension

    if shutil.which("xcrun") is None:
        raise EnvironmentError(
            "xcrun not found. Install Xcode Command Line Tools: "
            "xcode-select --install"
        )

    spec = importlib.util.spec_from_file_location(
        "gsplat_metal_build", os.path.join("gsplat", "metal", "build.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.compile_metallib()
    params = module.get_build_parameters()
    setup_dir = os.path.dirname(os.path.abspath(__file__))
    sources = [os.path.relpath(s, setup_dir) for s in params.sources]

    extension = CppExtension(
        params.name,
        sources=sources,
        include_dirs=params.include_dirs,
        extra_compile_args={"cxx": params.extra_cflags},
        extra_link_args=params.extra_ldflags,
    )
    return [extension]


ext_modules = []
if not BUILD_NO_CUDA:
    ext_modules.extend(get_extensions())
if BUILD_METAL:
    ext_modules.extend(get_metal_extensions())


setup(
    name="gsplat",
    version=__version__,
    description=" Python package for differentiable rasterization of gaussians",
    keywords="gaussian, splatting, cuda",
    url=URL,
    download_url=f"{URL}/archive/gsplat-{__version__}.tar.gz",
    python_requires=">=3.7",
    install_requires=[
        "ninja",
        "numpy",
        "jaxtyping",
        "rich>=12",
        "torch",
        "typing_extensions; python_version<'3.8'",
    ],
    extras_require={
        # lidar dependencies. Install them by `pip install gsplat[lidar]`
        "lidar": [
            "scipy",
        ],
        # CUDA-only runtime helpers. Install them by `pip install gsplat[cuda]`
        "cuda": [
            "cupy ; platform_system != 'Darwin'",
            "nerfacc>=0.5.3 ; platform_system != 'Darwin'",
        ],
        # Compression helpers. Install them by `pip install gsplat[compression]`
        "compression": [
            "PLAS @ git+https://github.com/fraunhoferhhi/PLAS.git ; platform_system != 'Darwin'",
            "torchpq>=0.3.0.6 ; platform_system != 'Darwin'",
            "cupy ; platform_system != 'Darwin'",
        ],
        # dev dependencies. Install them by `pip install gsplat[dev]`
        "dev": [
            "black[jupyter]==22.3.0",
            "isort==5.10.1",
            "pylint==2.13.4",
            "pytest==7.1.3",
            "pytest-env==0.8.1",
            "pytest-xdist==2.5.0",
            "typeguard>=2.13.3",
            "pyyaml>=6.0.1",
            "build",
            "twine",
            "imageio>=2.37.2",
        ],
    },
    ext_modules=ext_modules,
    cmdclass={"build_ext": get_ext()} if ext_modules else {},
    packages=find_packages(),
    package_data={"gsplat.metal": ["*.metallib"]},
    # https://github.com/pypa/setuptools/issues/1461#issuecomment-954725244
    include_package_data=True,
)
