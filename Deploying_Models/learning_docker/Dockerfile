# getting the base image
# If we do not want any base image
# SCRATCH is an empty image to build from scratch
# FROM SCRATCH

FROM ubuntu:latest

# Give who wants to maintain the image
# MAINTAINER aditya oke

############# NOTE #################################
# MAINTAINER has been deprecated
####################################################

# Running inside the docker image
# We have ubuntu so we can run the apt command
RUN apt-get update

# To run to the command line during the container creation use this
CMD [ "echo", "Hello World !!! From my first docker image" ]

# Difference between RUN and CMD
# RUN gets executed during building of the image
# CMD gets executed only when we create a container from the image.
