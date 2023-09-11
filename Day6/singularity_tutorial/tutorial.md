## Singularity tutorial

### 1. Load the module:

```
module load singularity
```

### 2. Execute a simple command with one singularity image.

```
$ singularity exec library://ubuntu:23.04 cat /etc/os-release
```
```
INFO:    Downloading library image
28.4MiB / 28.4MiB [========================================] 100 % 14.4 MiB/s 0s
INFO:    Converting SIF file to temporary sandbox...
WARNING: underlay of /etc/localtime required more than 50 (68) bind mounts
PRETTY_NAME="Ubuntu 22.04 LTS"
NAME="Ubuntu"
VERSION_ID="22.04"
VERSION="22.04 LTS (Jammy Jellyfish)"
VERSION_CODENAME=jammy
ID=ubuntu
ID_LIKE=debian
HOME_URL="https://www.ubuntu.com/"
SUPPORT_URL="https://help.ubuntu.com/"
BUG_REPORT_URL="https://bugs.launchpad.net/ubuntu/"
PRIVACY_POLICY_URL="https://www.ubuntu.com/legal/terms-and-policies/privacy-policy"
UBUNTU_CODENAME=jammy
INFO:    Cleaning up image...
```

In this command we pulled the image from the singularity image hub. You can also pull docker containers with Singularity. For instance:

```
$ singularity exec docker://ubuntu:22.04 cat /etc/os-release
```

### 3. Download Singularity images:    

```
$ singularity pull docker://ubuntu:22.04
...
$ ls
$ ubuntu_22.04.sif
```

You can use the downloaded image with:

```
$ singularity exec ./ubuntu_22.04.sif echo "Hello World"
```
### 4. Manage the image cache

When pulling images, Singularity stores images and blobs in a cache directory.

The default directory location for the image cache is ``$HOME/.singularity/cache``. You can redefine the path to the cache dir by setting the variable ``SINGULARITY_CACHEDIR``.

You can inspect the cache with:

```
$ singularity cache list -v

NAME                     DATE CREATED           SIZE             TYPE
34de800b5da88feb7723a8   2023-09-11 00:00:53    0.79 KiB         blob
445a6a12be2be54b4da18d   2023-09-11 00:03:30    28.17 MiB        blob
58690f9b18fca6469a14da   2023-09-11 00:00:51    44.34 MiB        blob
a19b95490ca5a60cd7d1fb   2023-09-11 00:00:53    2.42 KiB         blob
b492494d8e0113c4ad3fe4   2023-09-11 00:03:31    0.41 KiB         blob
b51569e7c50720acf68603   2023-09-11 00:00:52    0.84 KiB         blob
c6b84b685f35f1a5d63661   2023-09-11 00:03:31    2.25 KiB         blob
da8ef40b9ecabc2679fe24   2023-09-11 00:00:52    0.52 KiB         blob
fb15d46c38dcd1ea0b1990   2023-09-11 00:00:53    0.17 KiB         blob
sha256.3623d1371d291c9   2023-09-10 22:51:28    35.59 MiB        library
sha256.7a63c14842a5c9b   2023-09-10 22:56:58    28.44 MiB        library
1f1a2d56de1d604801a967   2023-09-11 00:00:57    35.59 MiB        oci-tmp
aabed3296a3d45cede1dc8   2023-09-11 00:03:33    28.45 MiB        oci-tmp
There are 4 container file(s) using 128.05 MiB and 9 oci blob file(s) using 72.52 MiB of space
Total space used: 200.57 MiB
```

If you want to wipe the cache, you can execute the command:

```
$ singularity cache clean
This will delete everything in your cache (containers from all sources and OCI blobs).
Hint: You can see exactly what would be deleted by canceling and using the --dry-run option.
Do you want to continue? [N/y] y
INFO:    Removing blob cache entry: blobs
INFO:    Removing blob cache entry: index.json
INFO:    Removing blob cache entry: oci-layout
INFO:    Removing library cache entry: sha256.3623d1371d291c97ca89c840922679f1aef8d4678ff3d617e85b5b175816af35
INFO:    Removing library cache entry: sha256.7a63c14842a5c9b9c0567c1530af87afbb82187444ea45fd7473726ca31a598b
INFO:    Removing oci-tmp cache entry: 1f1a2d56de1d604801a9671f301190704c25d604a416f59e03c04f5c6ffee0d6
INFO:    Removing oci-tmp cache entry: aabed3296a3d45cede1dc866a24476c4d7e093aa806263c27ddaadbdce3c1054
INFO:    No cached files to remove at /leonardo/home/userexternal/eposada0/.singularity/cache/shub
INFO:    No cached files to remove at /leonardo/home/userexternal/eposada0/.singularity/cache/oras
INFO:    No cached files to remove at /leonardo/home/userexternal/eposada0/.singularity/cache/net
```

Check of the cache was cleaned with ``singularity cache list -v``.

### 5. Files inside a container

Let us assess what the content of the root directory ``/`` looks like from outside vs inside the container. First, from the host system:

```
$ ls /
bin  boot  dev  etc  home  lib  lib64  media  mnt  opt  proc  root  run  sbin  scratch  shared  srv  sys  tmp  usr  var
```

Then, Inside the container:

```
$ singularity exec ./ubuntu_22.04.sif ls /
...
bin  boot  dev	environment  etc  home	leonardo  lib  lib32  lib64  libx32  media  mnt  opt  proc  root  run  sbin  singularity  srv  sys  tmp  usr  var
```

As you can see the container runs its own filesystem. By default the current $PATH is always bind-mounted. For example,

```
$ singularity exec ./ubuntu_22.04.sif pwd
/leonardo/home/userexternal/eposada0/singularity_tutorial/images
```

If you want to bind other directories you can add the flag ``-B`` or ``bind`` to the command line.

```
singularity -B dir1,dir2,dir3 ...
```

Equivalently, directories to be bind mounted can be specified using the environment variable ``SINGULARITY_BINDPATH``:

```
$ export SINGULARITY_BINDPATH="dir1,dir2,dir3"
```

### 6. Sharing environment with the HOST

By default, shell variables are inherited in the container from the host:

```
$ export HELLO=world
$ singularity exec ./ubuntu_22.04.sif bash -c 'echo $HELLO'
```
```
world
```

If on the contrary you want to avoid sharing the environment with the container, use ``-C``. (Note that this will also isolate system directories such as /tmp, /dev and /run)

```
$ export HELLO=world
$ singularity exec -C ./ubuntu_22.04.sif bash -c 'echo $HELLO'
```

If you need to pass only specific variables to the container, that might or might not be defined in the host, you can define variables with the SINGULARITYENV_ prefix:

```
$ export SINGULARITYENV_CIAO=mondo
$ singularity exec -C ./ubuntu_22.04.sif bash -c 'echo $CIAO'
```
```
mondo
```

An alternative way to define variables is to use the flag ``--env``:

```
$ singularity exec --env HOLA=mundo ./ubuntu_22.04.sif bash -c 'echo $HOLA'
```
```
mundo
```
#### Back to slides...

### 7. Build your own images 

Singularity can build container images in different formats. Let’s focus on the Singularity Image Format, ``SIF``, which is the one typically adopted to ship production-ready containers.

The definition file ``.def`` needed to build the image is:

```
$ cat def_files/lolcow.def
Bootstrap: docker
From: ubuntu:22.04

%post
    apt-get -y update
    apt-get -y install fortune cowsay lolcat

%environment
    export PATH=/usr/games:$PATH

%labels
    Author Gustavo Cerati
    Version v0.0.1

%help
    This is a demo container used to illustrate a def file.

%runscript
    fortune | cowsay | lolcat
```

To build the SIF image, we will use the command **(DO NOT EXECUTE ON THE CLUSTER!!!)**

```
sudo singularity build lolcow.sif lolcow.def
```
```
[sudo] password for fernando:
INFO:    Starting build...
Getting image source signatures
Copying blob 445a6a12be2b done
Copying config c6b84b685f done
Writing manifest to image destination
Storing signatures
...
INFO:    Adding help info
INFO:    Adding labels
INFO:    Adding environment to container
INFO:    Adding runscript
INFO:    Creating SIF file...
INFO:    Build complete: lolcow.sif
```

Let's test the image:

```
$ singularity exec lolcow.sif bash -c 'fortune | cowsay | lolcat'
INFO:    underlay of /etc/localtime required more than 50 (79) bind mounts
perl: warning: Setting locale failed.
perl: warning: Please check that your locale settings:
    LANGUAGE = (unset),
    LC_ALL = (unset),
    LANG = "en_US.UTF-8"
    are supported and installed on your system.
perl: warning: Falling back to the standard locale ("C").
 ________________________________
< Tomorrow, you can be anywhere. >
 --------------------------------
        \   ^__^
         \  (oo)\_______
            (__)\       )\/\
                ||----w |
                ||     ||
```

Admin rights are needed to build the image. If you want to build an image in a server where you do not have admin rights, Singularity offers the option to build the container remotely, using the Sylabs Remote Builder (https://cloud.sylabs.io/builder). You need an account and an access token.

Once you have the token, log with your credentials:

```
$ singularity remote login SylabsCloud
Generate an access token at https://cloud.sylabs.io/auth/tokens, and paste it here.
Token entered will be hidden for security.
Access Token:
INFO:    Access Token Verified!
INFO:    Token stored in /leonardo/home/userexternal/eposada0/.singularity/remote.yaml
```

Now, create the image:

```
singularity build -r lolcow_remote.sif lolcow.def
INFO:    Access Token Verified!
INFO:    Token stored in /root/.singularity/remote.yaml
INFO:    Remote "cloud.sylabs.io" now in use.
INFO:    Starting build...
...
INFO:    Creating SIF file...
INFO:    Build complete: /tmp/image-2256467157
INFO:    Performing post-build operations
INFO:    Format for SBOM is not set or file exceeds maximum size for SBOM generation.
INFO:    Calculating SIF image checksum
INFO:    Uploading image to library...
WARNING: Skipping container verification
INFO:    Uploading 87416832 bytes
INFO:    Image uploaded successfully.
INFO:    Build complete: lolcow_remote.sif
```

Check the SIF file is present:

```
$ ls
lolcow.def  lolcow_remote.sif
```

And test it:

```
$ singularity exec lolcow_remote.sif bash -c 'fortune | cowsay | lolcat'
INFO:    Converting SIF file to temporary sandbox...
WARNING: underlay of /etc/localtime required more than 50 (79) bind mounts
perl: warning: Setting locale failed.
perl: warning: Please check that your locale settings:
    LANGUAGE = (unset),
    LC_ALL = (unset),
    LANG = "en_US.UTF-8"
    are supported and installed on your system.
perl: warning: Falling back to the standard locale ("C").
 _________________________________________
/ Q: How do you shoot a blue elephant? A: \
| With a blue-elephant gun.               |
|                                         |
| Q: How do you shoot a pink elephant? A: |
| Twist its trunk until it turns blue,    |
| then shoot it with                      |
|                                         |
\ a blue-elephant gun.                    /
 -----------------------------------------
        \   ^__^
         \  (oo)\_______
            (__)\       )\/\
                ||----w |
                ||     ||
INFO:    Cleaning up image...
```

### 8. Run the container as an application

The ``%runscript`` section in the ``def`` file, allows you to define a default command for the image. For instance, in our example:

```
%runscript
    fortune | cowsay | lolcat
```

This command can then be used if you run the container as an executable:

```
$ ./lolcow_remote.sif
WARNING: underlay of /etc/localtime required more than 50 (79) bind mounts
perl: warning: Setting locale failed.
perl: warning: Please check that your locale settings:
    LANGUAGE = (unset),
    LC_ALL = (unset),
    LANG = "en_US.UTF-8"
    are supported and installed on your system.
perl: warning: Falling back to the standard locale ("C").
 ____________________________
/ Your lucky number has been \
\ disconnected.              /
 ----------------------------
        \   ^__^
         \  (oo)\_______
            (__)\       )\/\
                ||----w |
                ||     ||
```

Or, if you need to specify Singularity runtime flags, like ``-B``:
```
$ singularity run -B $HOME lolcow_remote.sif
INFO:    Converting SIF file to temporary sandbox...
WARNING: underlay of /etc/localtime required more than 50 (79) bind mounts
perl: warning: Setting locale failed.
perl: warning: Please check that your locale settings:
    LANGUAGE = (unset),
    LC_ALL = (unset),
    LANG = "en_US.UTF-8"
    are supported and installed on your system.
perl: warning: Falling back to the standard locale ("C").
______________________________________
< You are fairminded, just and loving. >
 --------------------------------------
        \   ^__^
         \  (oo)\_______
            (__)\       )\/\
                ||----w |
                ||     ||
INFO:    Cleaning up image...
```

You can also open a shell session in the container, for example to fix the ``perl`` warinings:

```
$ singularity shell lolcow_remote.sif
INFO:    Converting SIF file to temporary sandbox...
WARNING: underlay of /etc/localtime required more than 50 (79) bind mounts
Singularity> 
```
```
Singularity> export LC_ALL=C
Singularity> fortune | cowsay | lolcat
 _________________________________________
/ You would if you could but you can't so \
\ you won't.                              /
 -----------------------------------------
        \   ^__^
         \  (oo)\_______
            (__)\       )\/\
                ||----w |
                ||     ||
Singularity> exit
exit
INFO:    Cleaning up image...
```

### 9. Singularity and GPUs

In order to expose the GPUs to the Singularity container, use the ``--nv`` flag when running the singularity container. For instance, in a Compute node with GPU:

```
[lrdn3456 images]$ singularity run --nv ubuntu_22.04.sif nvidia-smi
INFO:    Converting SIF file to temporary sandbox...
WARNING: underlay of /etc/localtime required more than 50 (69) bind mounts
WARNING: underlay of /usr/bin/nvidia-smi required more than 50 (271) bind mounts
Mon Sep 11 08:15:20 2023
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 520.61.05    Driver Version: 520.61.05    CUDA Version: 11.8     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA PG506-243    On   | 00000000:1D:00.0 Off |                    0 |
| N/A   44C    P0    80W / 481W |      0MiB / 65536MiB |      0%      Default |
|                               |                      |             Disabled |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
INFO:    Cleaning up image...

```
And without the flag:

```
[@lrdn3456 images]$ singularity run  ubuntu_22.04.sif nvidia-smi
INFO:    Converting SIF file to temporary sandbox...
WARNING: underlay of /etc/localtime required more than 50 (69) bind mounts
FATAL:   "nvidia-smi": executable file not found in $PATH
INFO:    Cleaning up image...
```

## TASKS:

1. Fix the missing ``LC_ALL`` variable by exporting it into the ``def`` file and rebuilding the image again.

2. Change the ``%runscript section`` to accept arguments. (HINT: since it works like a script, you can also use variables like ``$1``, ``$2``, ``$@``, etc.)

3. Build a Python container with Pandas installed. The header of the definition file is:

```
Bootstrap: docker
From: python:latest
```

Pass the following script as argument when running the container:

```
# contents of hello_pandas.py
import numpy as np
import pandas as pd

df = pd.DataFrame({'A': 1.,
                   'B': pd.Timestamp('20130102'),
                   'C': pd.Series(1, index=list(range(4)), dtype='float32'),
                   'D': np.array([3] * 4, dtype='int32'),
                   'E': pd.Categorical(["test", "train", "test", "train"]),
                   'F': 'foo'})

print(df)
```

Compare your results with:

```
     A          B    C  D      E    F
0  1.0 2013-01-02  1.0  3   test  foo
1  1.0 2013-01-02  1.0  3  train  foo
2  1.0 2013-01-02  1.0  3   test  foo
3  1.0 2013-01-02  1.0  3  train  foo
```

4. Create and execute ``SLURM`` script to run a container and output the command ``nvidia-smi``.