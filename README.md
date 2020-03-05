# cloudray
ray tracing on the cloud

# env setup
prepare a windows destop, i.e. windows10

prepare a linux server,   i.e. ubuntu 16.04


# run x server and vnc server on linux server
apt install x11
export DISPLAY=:0
startx&
//only listen local VNC address for safety
x11vnc -noxdamage -display :0 -forever -scale 1600x1200 -listen 127.0.0.1 &
//sleep 3
nautilus &
//sleep 3
metacity &

# build on linux server
git clone https://github.com/woodspower/cloudray.git
cd cloudray/charter
c++ -std=c++11 -g cloudray.cpp -o cloudray

# run vnc client on windows desktop
install a 




