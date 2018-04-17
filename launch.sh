#! /bin/sh

REPO_DIR=`cd $(dirname $0); pwd`

docker run --rm -it -v $REPO_DIR:/work nkats/mln_gym
