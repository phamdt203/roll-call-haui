echo "[CHECKING] Checking if conda is present on the system..."

if ! command -v conda &> /dev/null
then
    echo "[INFO] Conda could not be found. Please install Miniconda or Anaconda first."
    exit
fi

# Ensure conda is initialized in the shell
eval "$(conda shell.bash hook)"

# Create or activate the conda environment
ENV_NAME="roll-call-haui"

# Check if the environment already exists
if conda env list | grep -q "$ENV_NAME"; then
    echo "[INFO] Activating existing environment: $ENV_NAME"
    conda activate "$ENV_NAME"
else
    echo "[INFO] Creating new environment: $ENV_NAME"
    conda create --name "$ENV_NAME" python=3.9 -y
    conda activate "$ENV_NAME"
fi

# Install the required packages
if [ -f "requirements.txt" ]; then
    echo "[INFO] Installing packages from requirements.txt"
    pip install -r requirements.txt
else
    echo "[INFO] requirements.txt not found, skipping package installation."
fi

# Print the status
echo "[INFO] Setup complete. Environment '$ENV_NAME' is activated."
