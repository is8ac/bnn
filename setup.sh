export TERM=xterm
sudo apt-get update
sudo apt-get install -y libfontconfig libfontconfig1-dev build-essential cmake git
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs -sSf | sh -s -- --default-toolchain nightly -y
source $HOME/.cargo/env
git clone https://github.com/is8ac/bnn.git
cd bnn
git checkout 388b3809879b611b21d1f9a474c183dc1a252c35
cargo build --release
