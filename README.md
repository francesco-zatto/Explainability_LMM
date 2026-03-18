# CoX-LMM: Concept eXplainability for Large Multimodal Models

A lightweight Jupyter Notebook implementation of the CoX-LMM framework. This project peeks inside the "black box" of Large Multimodal Models (LMMs) to extract internal representations and decompose them into interpretable "multimodal concepts" grounded in both vision and text.

## Features

- Model Analyzed: Uses LLaVA-1.5-7b loaded in 4-bit precision (via bitsandbytes) to run smoothly on consumer GPUs without out-of-memory errors.

- Dataset Automation: Automatically fetches and processes the MS COCO 2017 Validation split.

- Decomposition: Implements Semi-NMF (via scikit-learn) to extract sparse, non-negative concept activations from the LMM's deep residual stream.

- Visualization: Generates a clean matplotlib grid showing the Maximum Activating Samples (vision) alongside their top decoded tokens (text).

## Installation

Ensure you have Python installed, then install the required dependencies. (These are also included at the top of the notebook):

`
pip install torch torchvision transformers accelerate bitsandbytes scikit-learn datasets matplotlib pillow tqdm
`

## Usage

- Open CoX-LMM_Implementation.ipynb in Jupyter Notebook or JupyterLab.

- Run the cells sequentially.

- The notebook will automatically download the COCO dataset, load the LLaVA model, extract token representations, run Semi-NMF, and output the visualization grid.

- Change the target_token variable (e.g., from "dog" to "bus" or "cat") to explore how the model represents different concepts internally.

Note: For highly disentangled concepts, increase the sample size by removing the [:50] slice in the representation extraction loop!

## Credits

This implementation is based on the paper:

A Concept-Based Explainability Framework for Large Multimodal Models, Jayneel Parekh, Pegah Khayatan, Mustafa Shukor, Alasdair Newson, Matthieu Cord
