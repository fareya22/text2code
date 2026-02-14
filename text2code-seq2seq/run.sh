#!/bin/bash
# Convenient script to run the entire pipeline

set -e  # Exit on error

echo "=========================================="
echo "Text-to-Code Generation Pipeline"
echo "=========================================="
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt
echo "✓ Dependencies installed"

# Parse command line arguments
MODE=${1:-"all"}  # Default to "all" if no argument

case $MODE in
    "train")
        echo ""
        echo "=========================================="
        echo "TRAINING MODELS"
        echo "=========================================="
        python train.py ${2:-42}  # Default seed 42
        ;;
    
    "evaluate")
        echo ""
        echo "=========================================="
        echo "EVALUATING MODELS"
        echo "=========================================="
        python evaluate.py
        ;;
    
    "visualize")
        echo ""
        echo "=========================================="
        echo "VISUALIZING ATTENTION"
        echo "=========================================="
        python visualize_attention.py
        ;;
    
    "all")
        echo ""
        echo "=========================================="
        echo "STEP 1: TRAINING MODELS"
        echo "=========================================="
        python train.py ${2:-42}
        
        echo ""
        echo "=========================================="
        echo "STEP 2: EVALUATING MODELS"
        echo "=========================================="
        python evaluate.py
        
        echo ""
        echo "=========================================="
        echo "STEP 3: VISUALIZING ATTENTION"
        echo "=========================================="
        python visualize_attention.py
        
        echo ""
        echo "=========================================="
        echo "✓ PIPELINE COMPLETE!"
        echo "=========================================="
        echo ""
        echo "Results saved in:"
        echo "  - checkpoints/        (model weights)"
        echo "  - results/            (metrics & visualizations)"
        echo ""
        ;;
    
    *)
        echo "Usage: ./run.sh [MODE] [SEED]"
        echo ""
        echo "Modes:"
        echo "  train      - Train all models"
        echo "  evaluate   - Evaluate trained models"
        echo "  visualize  - Generate attention visualizations"
        echo "  all        - Run complete pipeline (default)"
        echo ""
        echo "Examples:"
        echo "  ./run.sh                  # Run everything with seed 42"
        echo "  ./run.sh all 123          # Run everything with seed 123"
        echo "  ./run.sh train 42         # Only train models"
        echo "  ./run.sh evaluate         # Only evaluate"
        echo "  ./run.sh visualize        # Only visualize attention"
        exit 1
        ;;
esac

echo ""
echo "Done!"
