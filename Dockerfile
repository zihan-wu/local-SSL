#####################################
# Image for local-SSL
#####################################
# We built the image based on available Nvidia images, which contains pytorch 2.0
# https://catalog.ngc.nvidia.com/orgs/nvidia/containers/

# IMPORTANT
# The --platform parameter here is linux/amd64. Adapt to your system
FROM nvcr.io/nvidia/pytorch:23.05-py3

# Install required packages
RUN apt update

#####################################
# For Users of EPFL RCP CaaS: Buildup
#####################################
# # Create your user inside the container.
# # This block is needed to correctly map
# # your EPFL user id inside the container.
# # Without this mapping, you are not able
# # to access files from the external storage.
# ARG LDAP_USERNAME
# ARG LDAP_UID
# ARG LDAP_GROUPNAME
# ARG LDAP_GID
# RUN groupadd ${LDAP_GROUPNAME} --gid ${LDAP_GID}
# RUN useradd -m -s /bin/bash -g ${LDAP_GROUPNAME} -u ${LDAP_UID} ${LDAP_USERNAME}

# # Copy your code inside the container
# RUN mkdir -p /home/${LDAP_USERNAME}
# COPY ./ /home/${LDAP_USERNAME}

# # Set your user as owner of the new copied files
# RUN chown -R ${LDAP_USERNAME}:${LDAP_GROUPNAME} /home/${LDAP_USERNAME}
# # Set the working directory in your user's home
# WORKDIR /home/${LDAP_USERNAME}
# USER ${LDAP_USERNAME}
# # PLEASE comment out the code in the block below
#####################################


#####################################
# For General Public
# Feel free to customize this section as needed
#####################################
# PLEASE make sure to comment out the above code for EPFL user
ARG LDAP_USERNAME
# Copy your code inside the container, make as working directory
RUN mkdir -p /home/${LDAP_USERNAME}
COPY ./ /home/${LDAP_USERNAME}
WORKDIR /home/${LDAP_USERNAME}
#####################################


# Install additional dependencies
RUN python -m pip install --upgrade pip
RUN pip install wandb pyyaml tqdm
