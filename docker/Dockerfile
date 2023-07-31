FROM ubuntu:20.04

WORKDIR /root/

# Add libcuda dummy dependency
ADD control .
RUN apt-get update && \
	DEBIAN_FRONTEND=noninteractive apt-get install --yes equivs && \
	equivs-build control && \
	dpkg -i libcuda1-dummy_11.8_all.deb && \
	rm control libcuda1-dummy_11.8* && \
	apt-get remove --yes --purge --autoremove equivs && \
	rm -rf /var/lib/apt/lists/*

# Setup Lambda repository
ADD lambda.gpg .
RUN apt-get update && \
	apt-get install --yes gnupg && \
	apt-key add lambda.gpg && \
	rm lambda.gpg && \
	echo "deb http://archive.lambdalabs.com/ubuntu focal main" > /etc/apt/sources.list.d/lambda.list && \
	echo "Package: *" > /etc/apt/preferences.d/lambda && \
	echo "Pin: origin archive.lambdalabs.com" >> /etc/apt/preferences.d/lambda && \
	echo "Pin-Priority: 1001" >> /etc/apt/preferences.d/lambda && \
	echo "cudnn cudnn/license_preseed select ACCEPT" | debconf-set-selections && \
	apt-get update && \
	DEBIAN_FRONTEND=noninteractive \
		apt-get install \
		--yes \
		--no-install-recommends \
		--option "Acquire::http::No-Cache=true" \
		--option "Acquire::http::Pipeline-Depth=0" \
		lambda-stack-cuda \
		lambda-server && \
	rm -rf /var/lib/apt/lists/*

# Setup for nvidia-docker
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV NVIDIA_REQUIRE_CUDA "cuda>=11.8"

COPY env.yml poetry.lock pyproject.toml poetry.toml add-pyenv-init.sh enable-pyenv-init-sh Makefile ./
RUN apt-get update && apt-get install -y build-essential cmake pkg-config gcc zlib1g-dev libbz2-dev libssl-dev libreadline-dev libsqlite3-dev libfreetype6-dev libblas-dev liblapack-dev gfortran wget curl libncurses5-dev xz-utils libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev jq zip unzip git
RUN make setup-pyenv-poetry
