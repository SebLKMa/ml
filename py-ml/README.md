# ML Coding
```sh
mkdir github.com
mkdir ml-coding
cd ml-coding
git init
```

Update **personal access key** in `.git/config`  
```sh
git pull origin master
```

## Pre-requisites
```sh
pip3 install numpy
pip3 install matplotlib
pip3 install seaborn
```

## 01 Linear Regression
Basic ML prediction of pizza orders using Linear Regression.  

## 02 Gradient Descent
Algorithm that can scale to the exponential combination of conditions and learning rate to minimize loss.

## 03 Matrix
Using matrix to handle multiple dimensions of input variables, e.g. Reservations, Temperature, Tourists etc. that affect outcome of Pizza orders.  

## References

https://pragprog.com/titles/pplearn/programming-machine-learning/


## Upgrading to Python 3.7

See also https://www.itsupportwale.com/blog/how-to-upgrade-to-python-3-7-on-ubuntu-18-10/  

```sh
ubuntu@ubuntu1804:~$ sudo apt-get install python3.7
Reading package lists... Done
Building dependency tree       
Reading state information... Done
The following packages were automatically installed and are no longer required:
  libev4 libuv1 libwebsockets12 linux-hwe-5.4-headers-5.4.0-64 linux-hwe-5.4-headers-5.4.0-65 linux-hwe-5.4-headers-5.4.0-66 linux-hwe-5.4-headers-5.4.0-67
  linux-hwe-5.4-headers-5.4.0-70 linux-hwe-5.4-headers-5.4.0-71 linux-hwe-5.4-headers-5.4.0-72
Use 'sudo apt autoremove' to remove them.
The following additional packages will be installed:
  libpython3.7-minimal libpython3.7-stdlib python3.7-minimal
Suggested packages:
  python3.7-venv python3.7-doc binfmt-support
The following NEW packages will be installed:
  libpython3.7-minimal libpython3.7-stdlib python3.7 python3.7-minimal
0 upgraded, 4 newly installed, 0 to remove and 8 not upgraded.
Need to get 4,281 kB of archives.
After this operation, 22.5 MB of additional disk space will be used.
Do you want to continue? [Y/n] Y
Get:1 http://us.archive.ubuntu.com/ubuntu bionic-updates/universe amd64 libpython3.7-minimal amd64 3.7.5-2~18.04.4 [546 kB]
Get:2 http://us.archive.ubuntu.com/ubuntu bionic-updates/universe amd64 python3.7-minimal amd64 3.7.5-2~18.04.4 [1,691 kB]
Get:3 http://us.archive.ubuntu.com/ubuntu bionic-updates/universe amd64 libpython3.7-stdlib amd64 3.7.5-2~18.04.4 [1,744 kB]
Get:4 http://us.archive.ubuntu.com/ubuntu bionic-updates/universe amd64 python3.7 amd64 3.7.5-2~18.04.4 [301 kB]
Fetched 4,281 kB in 5s (927 kB/s)   
Selecting previously unselected package libpython3.7-minimal:amd64.
(Reading database ... 421177 files and directories currently installed.)
Preparing to unpack .../libpython3.7-minimal_3.7.5-2~18.04.4_amd64.deb ...
Unpacking libpython3.7-minimal:amd64 (3.7.5-2~18.04.4) ...
Selecting previously unselected package python3.7-minimal.
Preparing to unpack .../python3.7-minimal_3.7.5-2~18.04.4_amd64.deb ...
Unpacking python3.7-minimal (3.7.5-2~18.04.4) ...
Selecting previously unselected package libpython3.7-stdlib:amd64.
Preparing to unpack .../libpython3.7-stdlib_3.7.5-2~18.04.4_amd64.deb ...
Unpacking libpython3.7-stdlib:amd64 (3.7.5-2~18.04.4) ...
Selecting previously unselected package python3.7.
Preparing to unpack .../python3.7_3.7.5-2~18.04.4_amd64.deb ...
Unpacking python3.7 (3.7.5-2~18.04.4) ...
Setting up libpython3.7-minimal:amd64 (3.7.5-2~18.04.4) ...
Setting up python3.7-minimal (3.7.5-2~18.04.4) ...
Setting up libpython3.7-stdlib:amd64 (3.7.5-2~18.04.4) ...
Setting up python3.7 (3.7.5-2~18.04.4) ...
Processing triggers for gnome-menus (3.13.3-11ubuntu1.1) ...
Processing triggers for mime-support (3.60ubuntu1) ...
Processing triggers for desktop-file-utils (0.23-1ubuntu3.18.04.2) ...
Processing triggers for man-db (2.8.3-2ubuntu0.1) ...
ubuntu@ubuntu1804:~$ 
ubuntu@ubuntu1804:~$ 
ubuntu@ubuntu1804:~$ python3 -V
Python 3.6.9
ubuntu@ubuntu1804:~$ sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.6 1
update-alternatives: using /usr/bin/python3.6 to provide /usr/bin/python3 (python3) in auto mode
ubuntu@ubuntu1804:~$ sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 2
update-alternatives: using /usr/bin/python3.7 to provide /usr/bin/python3 (python3) in auto mode
ubuntu@ubuntu1804:~$ sudo update-alternatives --config python3
There are 2 choices for the alternative python3 (providing /usr/bin/python3).

  Selection    Path                Priority   Status
------------------------------------------------------------
* 0            /usr/bin/python3.7   2         auto mode
  1            /usr/bin/python3.6   1         manual mode
  2            /usr/bin/python3.7   2         manual mode

Press <enter> to keep the current choice[*], or type selection number: 2
ubuntu@ubuntu1804:~$ python3 -V
Python 3.7.5

```