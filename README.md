# chatgpt_grounding
## Overview
This repository provides a framework for evaluating the alignment of conceptual representation between Large Language Models (LLMs) and humans, based on the findings from the paper Large Language Models without Grounding Recover Non-Sensorimotor but Not Sensorimotor Features of Human Concepts ([arXiv link]). Our analysis investigates how well LLMs capture different dimensions of human conceptual knowledge, distinguishing between non-sensorimotor, sensory, and motor domains.

## Key Features
- Reproducible Evaluation Pipeline: We provide a Python-based pipeline that enables researchers to systematically assess various LLMs on their conceptual grounding.
- Continuous Updates: As new models emerge, we will continue updating the repository to keep it relevant for the research community.
- Recent Analysis: We have recently evaluated DeepSeek, a new LLM that has gained significant attention, using our pipeline. The results were consistent with our previous findings on ChatGPT and Google LLMs. The evaluation is included in the repository as an example.
- Dimension-wise Analysis: The repository currently supports a dimension-wise analysis of conceptual representations in LLMs.

## Requirements
**Python version**: 3.11.7<br>
**Dependencies**:  Ensure you have the required packages installed by running:
  ```bash
  pip install -r requirements.txt
  ```

## Usage
To run the dimension-wise analysis, execute the following command:
```bash
python scripts/aggregated.py \
  --human_aggregated_directory example_files/human \
  --model_directory example_files/model \
  --output_path example_files/test_output.csv
```
Note that 

## Arguments
--human_aggregated_directory: Directory containing human-annotated concept data.<br>
--model_directory: Directory containing model-generated representations.<br>
--output_path: Path to save the analysis output.<br>

## Planned Updates
TODO

## Contributions
We welcome contributions from the research community! If you have suggestions, feature requests, or bug reports, please open an issue or submit a pull request.

## Citation
If you use this repository in your research, please cite our paper:

```bibtex
@article{YourCitationHere,
  author = {Authors},
  title = {Large Language Models without Grounding Recover Non-Sensorimotor but Not Sensorimotor Features of Human Concepts},
  journal = {arXiv},
  year = {202X},
  url = {your-arxiv-link-here}
}
```
## Contact
For any questions or collaboration inquiries, please reach out via [qihuixu01@gmail.com].




