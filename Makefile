IMAGE ?= sandai/magi-compiler:latest
DOCKERFILE ?= Dockerfile

MAGIDIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))

.PHONY: docker-build docker-push

# make docker-build
docker-build:
	DOCKER_BUILDKIT=1 docker build \
		--secret id=http_proxy,env=http_proxy \
		--secret id=https_proxy,env=https_proxy \
		-f $(MAGIDIR)/$(DOCKERFILE) \
		-t $(IMAGE) \
		$(MAGIDIR)

# make docker-push
docker-push:
	docker push $(IMAGE)
