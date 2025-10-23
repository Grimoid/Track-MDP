# Contributing to Track-MDP

We welcome contributions to the Track-MDP project! This document provides guidelines for contributing.

## Quick Start for Contributors

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/yourusername/Track-MDP-final.git
   cd Track-MDP-final
   ```
3. **Create a new branch** for your feature:
   ```bash
   git checkout -b feature/your-feature-name
   ```
4. **Make your changes** and test them
5. **Submit a pull request**

## Development Setup

### Prerequisites

- Python 3.8+
- Git
- Virtual environment tool (venv, conda, etc.)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/Track-MDP-final.git
cd Track-MDP-final

# Create virtual environment
python -m venv track_mdp_dev
source track_mdp_dev/bin/activate  # On Windows: track_mdp_dev\Scripts\activate

# Install development dependencies
pip install -r requirements.txt
pip install -e .

# Install development tools
pip install pytest black flake8 mypy pre-commit

# Set up pre-commit hooks
pre-commit install
```

## Types of Contributions

### Bug Reports

When reporting bugs, please include:

- **Environment details**: OS, Python version, dependency versions
- **Minimal reproduction case**: Simplest code that reproduces the issue
- **Expected vs actual behavior**: What you expected vs what happened
- **Error messages**: Full stack trace if applicable

Use the bug report template in GitHub Issues.

### Feature Requests

For new features, please:

- **Check existing issues** to avoid duplicates
- **Describe the use case** clearly
- **Propose an implementation** approach if possible
- **Consider backwards compatibility**

### Code Contributions

#### Areas for Contribution

1. **Core Environment**
   - New environment configurations
   - Performance optimizations
   - Additional reward structures

2. **Training Algorithms**
   - Support for new RL algorithms (PPO, SAC, etc.)
   - Hyperparameter optimization
   - Distributed training improvements

3. **Evaluation & Visualization**
   - New evaluation metrics
   - Enhanced visualization features
   - Performance analysis tools

4. **Documentation**
   - Code documentation improvements
   - Tutorial additions
   - Example enhancements

5. **Testing**
   - Unit test coverage
   - Integration tests
   - Performance benchmarks

## Code Style Guidelines

### Python Code Style

We follow PEP 8 with some modifications:

- **Line length**: 88 characters (Black default)
- **Imports**: Use isort for import sorting
- **Type hints**: Required for new functions
- **Docstrings**: Google style for all public functions

### Pre-commit Hooks

We use pre-commit hooks to ensure code quality:

```bash
# Automatically run on commit
git commit -m "Your commit message"

# Run manually
pre-commit run --all-files
```

The hooks include:
- **Black**: Code formatting
- **Flake8**: Linting
- **MyPy**: Type checking
- **isort**: Import sorting

### Code Formatting Example

```python
"""
Module docstring explaining the purpose.

This module implements object tracking functionality for reinforcement
learning agents in grid environments.
"""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from ray.rllib.algorithms.a2c import A2C


class TrackingAgent:
    """
    A reinforcement learning agent for object tracking.
    
    This class provides functionality for training and evaluating
    tracking policies using various RL algorithms.
    
    Args:
        grid_size: Size of the NxN tracking grid
        algorithm: RL algorithm name ('a2c', 'ppo', etc.)
        config: Algorithm-specific configuration dictionary
    
    Example:
        >>> agent = TrackingAgent(grid_size=10, algorithm='a2c')
        >>> agent.train(num_iterations=1000)
        >>> accuracy = agent.evaluate()
    """
    
    def __init__(
        self, 
        grid_size: int, 
        algorithm: str = 'a2c',
        config: Optional[Dict] = None
    ) -> None:
        self.grid_size = grid_size
        self.algorithm = algorithm
        self.config = config or {}
        
        # Initialize components
        self._setup_environment()
        self._setup_algorithm()
    
    def train(self, num_iterations: int) -> Dict[str, float]:
        """
        Train the tracking agent.
        
        Args:
            num_iterations: Number of training iterations
            
        Returns:
            Training metrics dictionary with keys:
            - 'final_reward': Final episode reward mean
            - 'training_time': Total training time in seconds
            
        Raises:
            ValueError: If num_iterations is not positive
        """
        if num_iterations <= 0:
            raise ValueError("num_iterations must be positive")
        
        # Training implementation
        # ...
        
        return {
            'final_reward': 0.0,
            'training_time': 0.0
        }
```

## Testing Guidelines

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_environment.py

# Run with verbose output
pytest -v
```

### Writing Tests

#### Unit Tests

```python
import pytest
import numpy as np
from src.core.environment import learning_grid_sarsa_0


class TestEnvironment:
    """Test cases for the grid environment."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.env = learning_grid_sarsa_0(
            run_number=999,
            N=5,
            num_trans=3,
            state_trans_cum_prob=[0.4, 0.7, 1.0],
            max_sensors=3,
            max_sensors_null=3,
            time_limit=1,
            time_limit_max=1
        )
    
    def test_environment_creation(self):
        """Test that environment is created with correct parameters."""
        assert self.env.N == 5
        assert self.env.num_trans == 3
        assert self.env.time_limit == 1
    
    def test_object_movement(self):
        """Test that object moves correctly."""
        initial_pos = self.env.grid_env.object_pos
        self.env.grid_env.object_move()
        new_pos = self.env.grid_env.object_pos
        
        # Object should either move or reach terminal state
        assert new_pos != initial_pos or new_pos == self.env.N * self.env.N
    
    @pytest.mark.parametrize("grid_size", [3, 5, 8, 10])
    def test_multiple_grid_sizes(self, grid_size):
        """Test environment with different grid sizes."""
        env = learning_grid_sarsa_0(
            run_number=999, N=grid_size, num_trans=2,
            state_trans_cum_prob=[0.5, 1.0], max_sensors=3,
            max_sensors_null=3, time_limit=1, time_limit_max=1
        )
        assert env.N == grid_size
        assert env.grid_env.object_pos < grid_size * grid_size
```

#### Integration Tests

```python
def test_training_integration():
    """Test complete training workflow."""
    from src.training.trainer import train
    
    # Run minimal training for testing
    result = train(
        visualization_mode='none',
        num_iterations=5,  # Very short for testing
        test_mode=True
    )
    
    assert result is not None
    assert 'final_accuracy' in result
    assert result['final_accuracy'] >= 0.0
```

### Performance Tests

```python
import time
import pytest


@pytest.mark.performance
def test_environment_performance():
    """Test that environment operations complete within reasonable time."""
    env = create_test_environment()
    
    start_time = time.time()
    
    # Run 1000 object movements
    for _ in range(1000):
        env.grid_env.object_move()
        if env.grid_env.object_pos == env.N * env.N:
            env.grid_env.reset_object_state()
    
    elapsed = time.time() - start_time
    
    # Should complete in under 1 second
    assert elapsed < 1.0, f"Performance test took {elapsed:.2f}s, expected < 1.0s"
```

## Documentation Guidelines

### Code Documentation

- **All public functions** must have docstrings
- **Complex algorithms** need inline comments
- **Type hints** required for new code
- **Examples** in docstrings when helpful

### Documentation Updates

When adding features, update:

1. **Docstrings** in the code
2. **README.md** if adding new capabilities
3. **Examples** if demonstrating new features
4. **API documentation** for public interfaces

## Pull Request Process

### Before Submitting

1. **Run tests**: Ensure all tests pass
2. **Run linting**: Fix any style issues
3. **Update docs**: Add/update documentation
4. **Add tests**: Include tests for new features
5. **Check performance**: Ensure no significant slowdowns

### PR Description Template

```markdown
## Summary
Brief description of changes

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Testing
- [ ] Tests pass locally
- [ ] Added new tests for changes
- [ ] Checked performance impact

## Documentation
- [ ] Updated code documentation
- [ ] Updated README if needed
- [ ] Added examples if needed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Commented complex code
- [ ] No unnecessary debug prints
```

### Review Process

1. **Automated checks** must pass (CI/CD)
2. **Code review** by maintainer
3. **Testing** on different platforms if needed
4. **Documentation review**
5. **Merge** after approval

## Release Process

### Version Numbers

We follow [Semantic Versioning](https://semver.org/):

- **MAJOR**: Incompatible API changes
- **MINOR**: Backwards-compatible functionality additions
- **PATCH**: Backwards-compatible bug fixes

### Release Checklist

1. Update version in `src/__init__.py`
2. Update `CHANGELOG.md`
3. Run full test suite
4. Create release tag
5. Build and upload package
6. Update documentation

## Getting Help

### Communication Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and ideas
- **Pull Requests**: Code contributions

### Code of Conduct

We follow the [Contributor Covenant](https://www.contributor-covenant.org/version/2/1/code_of_conduct/):

- **Be respectful** and inclusive
- **Be patient** with new contributors
- **Be constructive** in feedback
- **Focus on the work**, not the person

### Mentorship

New contributors are welcome! If you're new to:

- **Open source**: Check our "good first issue" labels
- **Machine learning**: See our documentation and examples
- **Ray RLlib**: We can help with integration questions

Thank you for contributing to Track-MDP! 🎯