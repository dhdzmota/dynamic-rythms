#  Project Summary: dynamic-rythms

This is a repo developed to solve the _Dynamic Rhythms Contest_, focused on predicting power outage caused by weather 
events, specially storms.
![](assets/images/dynamic-rhythms.png)

## Objective: 
Develop a model that can forecast power outages in advance based on historical and real-time meteorological data. 
The goal is to enable early, accurate, and geospatially precise predictions that support proactive energy system responses.


## Outcome: 
By participating, this project will contribute to the advancement of rare-event forecasting and support the development
of more sustainable and resilient energy systems through improved outage prediction.
---

## Steps to Run This Project

Follow these steps to set up and run the pipeline end to end:

---

### 1. Prepare the Data
After cloning or downloading this repository, you need to manually place the contest data into the appropriate folder:
- Unzip the `dynamic-rhythms-train-data` archive into the `data/raw` directory.
- The extracted folder **must** be named exactly `dynamic-rhythms-train-data`.

Your folder structure should look like this:
```
    dynamic-rythms/
    ├── data/
    │   ├── external/                        # Data from third-party sources.
    │   ├── final/                           # Final version of data, ready for model, and test.
    │   ├── temp_results/                    # Temporary files that are generated from results.
    │   ├── interim/                         # Intermediate data files.
    │   └── raw/                             # Original, immutable data.
    │       └── dynamic-rhythms-train-data/  # Provided by the contest.
    │           └── data/
    │               ├── eaglei_data/
    │               └── NOAA_StormEvents/
    │       
```
- The expected path is: `data/raw/dynamic-rhythms-train-data/`

### 2. Create a Virtual Environment

Make sure you're using **Python version 3.10.17**.
Then, create and activate a virtual environment using your preferred method.  


### 3. Install Project Dependencies

From the project root, install the required packages: `pip install -r requirements.txt`


### 4. Run the Pipeline
You can choose between running the notebook or using the script:
- Option A: Use Jupyter Notebook
  - Open and run the notebook `model_pipeline.ipynb` cell by cell. This will download external data, generate features, train the model and evaluate it.
- Option B: Run the Pipeline script
  - Alternative you can run the entire pipeline via script `python general_pipeline.py`. This will perform all the steps except for model evaluation.
Once it completes, you can open the notebook at step 4 to inspect the results or continue evaluating.

 ### 5. Sit back and let it run
The pipeline will process the data, build features, train the model, and provide outputs.
Depending on your machine and internet speed, this may take some time.

---
## License
This project is licensed under the [MIT License](./LICENSE).  

## Contributors

- [Daniel Hernández Mota](https://github.com/dhdzmota)
- [Kim Alejandro Mora Trujillo](https://github.com/Kim-Mora)
