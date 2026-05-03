.PHONY: env build-metal test-metal clean-metal metallib

env:
	conda env create -f environment.yml
	@echo "Run: conda activate gsplat-metal"
	@echo "Then run: make build-metal"

build-metal:
	KMP_DUPLICATE_LIB_OK=TRUE BUILD_NO_CUDA=1 pip install -e . --no-build-isolation

metallib:
	bash scripts/build_metal.sh

test-metal:
	pytest tests/metal/ -v

clean-metal:
	find gsplat/metal/csrc -name "*.air" -delete
	rm -f gsplat/metal/gsplat_metal.metallib
	rm -f gsplat/metal_ext*.so
