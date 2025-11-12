## LLMs from Scratch Assignment 2

#### Environments

- Configure the Python environment by running `pip install -r requirements.txt`.

#### Testing

- To run all tests, use `pytest`.
- To run a specific test script, use `pytest src/test_{your_test_name}.py`. Make sure to run this command from the project root directory (do not cd into src/).
- To run a specific test function, use `pytest -k test_{your_test_name}`. Don't include `src/` or `.py` here!
- `pytest` will capture all outputs by default when the test succeeds. To disable output capturing (and see your debug outputs), use the flag `-s`.


#### Submission

- Submit a ZIP file `{your_name}_{your_student_id}.zip`, for example, `xinghan_li_2022012747.zip`. Your submission should contain the following files:

```
{your_name}_{your_student_id}.zip/
├── src/
│   ├── ... (your implementation files for Tasks 1–3)
├── ... (your implementation files for Task 4)
├── README.md  (instructions on how to reproduce your experiments in Task 4 step by step)
└── writeup.pdf  (your experiment report, including checkpoint paths, training and validation curves, evaluation accuracies on all benchmarks, and other relevant results)
```


If you have implemented the Bonus Challenge, submit your code and results to us by email. Check: https://git.tsinghua.edu.cn/tangkx25/llm_training_with_constrained_compute/
