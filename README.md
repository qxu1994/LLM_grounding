# chatgpt_grounding
## Overview
This repository provides a framework for evaluating the alignment of conceptual representation between Large Language Models (LLMs) and humans, based on the findings from the paper Large Language Models without Grounding Recover Non-Sensorimotor but Not Sensorimotor Features of Human Concepts ([arXiv link]). Our analysis investigates how well LLMs capture different dimensions of human conceptual knowledge, distinguishing between non-sensorimotor, sensory, and motor domains.

## Key Features
- Reproducible Evaluation Pipeline: We provide a Python-based pipeline that enables researchers to systematically assess various LLMs on their conceptual grounding.
- Continuous Updates: As new models emerge, we will continue updating the repository to keep it relevant for the research community.
- Recent Analysis: We have recently evaluated DeepSeek, a new LLM that has gained significant attention, using our pipeline. The results were consistent with our previous findings on ChatGPT and Google LLMs. The evaluation is included in the repository as an example.
- Dimension-wise Analysis: The repository currently supports a dimension-wise analysis of conceptual representations in LLMs.

## Current progress



## How to use the pipeline
**Python version**: 3.11.7<br>
**Dependencies**:  Ensure you have the required packages installed by running:
  ```bash
  pip install -r requirements.txt
  ```

## Usage
To run the dimension-wise analysis, execute the following command:
```bash
python scripts/aggregated.py \
  --human_aggregated_directory scripts/example_files/human \
  --model_directory scripts/example_files/model \
  --output_path scripts/example_files/test_output.csv
```
### Human data
The pipeline uses word-level human data
- For the Glasgow norms, the word-level data is accessible at [https://doi.org/10.3758/s13428-018-1099-3].
- For the Lancaster norms, the word-level data is accessible at [https://embodiedcognitionlab.shinyapps.io/sensorimotor_norms/]
- Make sure that in the directory pointing to human data, **the Glasgow norms should be renamed to glasgow_human.csv and the Lancaster norms should be renamed to lancaster_human.csv.**
### Model responses
- Model responses should be organized into separate CSV files, with one file per model. Glasgow and Lancaster responses should be saved in separate files.
- **File names must follow the format {dataset}_{model}.csv**, where dataset should be either *glasgow* or *lancaster*, and model should be the name of your model.
- Please refer to example_files/model for example files.

### Arguments
--human_aggregated_directory: Directory containing human-annotated concept data.<br>
--model_directory: Directory containing model-generated representations.<br>
--output_path: Path to save the analysis output.<br>
### Output
- the pipeline will output a csv file and a png file.
- The csv file contains model-human correlations (spearman coefficient, 95% CI, and significance value) for each model and dimension.
- The png file presents dimension-wise model-human correlations separately for each model.

## Planned Updates
- Simplify the format of input files

## Contributions
We welcome contributions from the research community! If you have suggestions, feature requests, or bug reports, please open an issue or submit a pull request.

## Citation
If you use this repository in your research, please cite our paper:

```bibtex
@article{llmwithoutgrounding,
  author = {Qihui Xu, Yingying Peng, Samuel Nastase, Martin Chodorow, Minghua Wu, Ping Li},
  title = {Large Language Models without Grounding Recover Non-Sensorimotor but Not Sensorimotor Features of Human Concepts},
  journal = {arXiv},
  year = {2025},
  url = {your-arxiv-link-here}
}
```
## Contact
For any questions or collaboration inquiries, please reach out via [qihuixu01@gmail.com].




