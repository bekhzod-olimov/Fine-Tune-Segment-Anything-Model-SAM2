# Define the URLs for the checkpoints
BASE_URL="https://dl.fbaipublicfiles.com/segment_anything_2/072824/"
sam2_hiera_t_url="${BASE_URL}sam2_hiera_tiny.pt"
sam2_hiera_s_url="${BASE_URL}sam2_hiera_small.pt"
sam2_hiera_b_plus_url="${BASE_URL}sam2_hiera_base_plus.pt"
sam2_hiera_l_url="${BASE_URL}sam2_hiera_large.pt"

# Create the checkpoints directory if it doesn't exist

root_dir="backup"
save_dir="${root_dir}/checkpoints"
cd ../
mkdir -p ${root_dir}
mkdir -p ${save_dir}
cd Fine-Tune-Segment-Anything-Model-SAM2

# Download each of the four checkpoints using wget
echo "Downloading sam2_hiera_tiny.pt checkpoint..."
wget -P ../${save_dir} $sam2_hiera_t_url || { echo "Failed to download checkpoint from $sam2_hiera_t_url"; exit 1; }

echo "Downloading sam2_hiera_small.pt checkpoint..."
wget -P ../${save_dir} $sam2_hiera_s_url || { echo "Failed to download checkpoint from $sam2_hiera_s_url"; exit 1; }

echo "Downloading sam2_hiera_base_plus.pt checkpoint..."
wget -P ../${save_dir} $sam2_hiera_b_plus_url || { echo "Failed to download checkpoint from $sam2_hiera_b_plus_url"; exit 1; }

echo "Downloading sam2_hiera_large.pt checkpoint..."
wget -P ../${save_dir} $sam2_hiera_l_url || { echo "Failed to download checkpoint from $sam2_hiera_l_url"; exit 1; }

echo "All checkpoints are downloaded successfully."