export TERM=xterm
sudo apt-get update
sudo apt-get install -y libfontconfig libfontconfig1-dev build-essential cmake git
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs -sSf | sh -s -- --default-toolchain nightly -y
source $HOME/.cargo/env
git clone https://github.com/is8ac/bnn.git
cd bnn
git checkout 0f58541a4a7e997c96b459b666f2c657770eaf9a
cargo build --release
