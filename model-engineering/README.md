# Model Engineering

This layer converts the approved datasets into a compact MUFASA model, evaluates scientific reasoning and grounding, and packages the accepted checkpoint as a `llama.cpp`-compatible GGUF.

![MUFASA model-training pipeline](./images/model-training-pipeline.svg)

See the [full model-training pipeline](./model-training-pipeline.md) for the training stages, evaluation gates and deployment decisions.

Model weights and training checkpoints are release artifacts and are not stored in Git.
