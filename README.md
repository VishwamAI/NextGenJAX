# NextGenJAX
[![Python package](https://github.com/VishwamAI/NextGenJAX/actions/workflows/ci.yml/badge.svg)](https://github.com/VishwamAI/NextGenJAX/actions/workflows/ci.yml)

## Overview
NextGenJAX is an advanced neural network library built on top of JAX, designed to surpass the capabilities of existing libraries such as Google DeepMind's Haiku and Optax. It leverages the flexibility and performance of JAX and Flax to provide a modular, high-performance, and easy-to-use framework for building and training neural networks.

## Framework Compatibility
NextGenJAX now supports both TensorFlow and PyTorch, allowing users to choose their preferred deep learning framework. This compatibility enables seamless integration with existing TensorFlow or PyTorch workflows while leveraging the advanced features of NextGenJAX.

## Features
- Modular design with customizable layers and activation functions
- Support for various optimizers, including custom optimizers
- Flexible training loop with support for custom loss functions
- Integration with JAX and Flax for high performance and scalability
- Comprehensive test suite to ensure model correctness and performance

## Installation
To install NextGenJAX, you can use pip:
```bash
pip install nextgenjax
```

For development, clone the repository and install the required dependencies:
```bash
git clone https://github.com/VishwamAI/NextGenJAX.git
cd NextGenJAX
pip install -r requirements.txt
```

NextGenJAX now supports both TensorFlow and PyTorch. To use these frameworks, make sure to install them separately:

For TensorFlow:
```bash
pip install tensorflow>=2.9.1
```

For PyTorch:
```bash
pip install torch>=1.12.1
```

## Usage
NextGenJAX now supports both TensorFlow and PyTorch frameworks. Users can choose their preferred framework when initializing the model.

### Creating a Model
To create a model using NextGenJAX, choose your framework and initialize the model:

```python
from src.model import NextGenModel

# Initialize the model with TensorFlow
tf_model = NextGenModel(framework='tensorflow', num_layers=6, hidden_size=512, num_heads=8, dropout_rate=0.1)

# Initialize the model with PyTorch
pytorch_model = NextGenModel(framework='pytorch', num_layers=6, hidden_size=512, num_heads=8, dropout_rate=0.1)
```

### Training the Model
The training process remains similar for both frameworks. Here's an example using TensorFlow:

```python
import tensorflow as tf
from src.train import create_train_state, train_model

# Define the optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# Create the training state
train_state = create_train_state(tf_model, optimizer)

# Define the training data and loss function
train_data = ...  # Your training data here
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

# Train the model
train_model(train_state, train_data, loss_fn, num_epochs=10)
```

For PyTorch, the process is similar, but you'll use PyTorch-specific optimizers and loss functions.

Note: The core functionality remains the same for both frameworks, allowing users to leverage either TensorFlow or PyTorch based on their preference or specific use case.

## Development Setup
To set up a development environment:

1. Clone the repository
2. Install development dependencies: `pip install -r requirements-dev.txt`
3. Run tests using pytest: `pytest tests/`

We use GitHub Actions for continuous integration and deployment. Our CI/CD workflow runs tests on Python 3.9 to ensure compatibility and code quality.

## Community and Support

We welcome community engagement and support for the NextGenJAX project:

- **Discussions**: Join our community discussions at [NextGenJAX Discussions](https://github.com/VishwamAI/NextGenJAX/discussions)
- **Contact**: For additional support or inquiries, you can reach us at [aivishwam@gmail.com](mailto:aivishwam@gmail.com)

## Contributing
We welcome contributions to NextGenJAX! Please follow these steps:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/your-feature`)
3. Make your changes and commit them (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Create a new pull request using the [Pull Request Template](.github/PULL_REQUEST_TEMPLATE.md)

Please adhere to our coding standards:
- Follow PEP 8 guidelines
- Write unit tests for new features
- Update documentation as necessary

For more detailed guidelines, please refer to the [CONTRIBUTING.md](CONTRIBUTING.md) file.

## Reporting Issues
If you encounter any issues or have suggestions for improvements, please open an issue in the repository. Use the appropriate issue template:
- [Bug Report](.github/ISSUE_TEMPLATE/bug_report.md)
- [Feature Request](.github/ISSUE_TEMPLATE/feature_request.md)

Provide as much detail as possible to help us understand and address the problem.

## License
NextGenJAX is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.

## Acknowledgements
NextGenJAX is inspired by the work of Google DeepMind and the JAX and Flax communities. We thank them for their contributions to the field of machine learning.

## Contact Information
For support or questions about NextGenJAX, please reach out to:

- Email: [aivishwam@gmail.com](mailto:aivishwam@gmail.com)
- GitHub Issues: [NextGenJAX Issues](https://github.com/VishwamAI/NextGenJAX/issues)
- Community Forum: [NextGenJAX Discussions](https://github.com/VishwamAI/NextGenJAX/discussions)

Last updated: 2023-05-10 12:00:00 UTC
