# ukbFound

This is the official codebase for **A foundational model encodes deep phenotyping data and enables diverse downstream application**. 

ukbFound: a foundation model with 25.3 million parameters that encoded thousands of individual-level traits into language-like sequences. By incorporating domain-specific tokenization, position-free embedding, and interpretable reasoning, ukbFound effectively captures latent disease-trait relationships from deep phenotyping data of 502,342 UK Biobank individuals.


## Main Features

- UK Biobank data preprocessing and feature extraction
- ukbFound pretraining
- Interpretability analysis

## Quick Start

```python
import ukbfound as ukf

# Load data
data = ukf.load_ukb_data("path/to/your/data")

# Preprocess
processed_data = ukf.preprocess(data)

# Train model
model = ukf.train_model(processed_data)

# Evaluate
results = ukf.evaluate(model, processed_data)
```


## Acknowledgements

We sincerely thank the authors of following open-source projects:

- [scGPT](https://github.com/bowang-lab/scGPT) - For the inspiration of transformer architecture and implementation details



## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 

