import datetime
import meds
import datasets
import numpy as np
from femr.model_utils import get_model_vocab

### MEDS v1 version
# def get_patient_birthdate(patient: meds.subject_id_dtype) -> datetime.datetime:
#     for e in patient["events"]:
#         for m in e["measurements"]:
#             if m["code"] == meds.birth_code:
#                 return e["time"]
#     raise ValueError("Couldn't find patient birthdate -- Patient has no events " + str(patient["events"][:5]))


### MEDS v3 version
def get_patient_birthdate(patient: datasets.Dataset) -> datetime.datetime:
    """
    Get patient birthdate from a flat dataset format.
    
    Args:
        patient: A row from the dataset containing patient information
        
    Returns:
        datetime.datetime: The patient's birthdate

    Example:
        >>> patient = MEDSV3_DATA.filter(lambda x: x['subject_id'] == 101)
        >>> get_patient_birthdate(patient)
    """
    # Check if birthdate is directly available in the dataset
    for i, code in enumerate(patient['code']):
        if code == meds.birth_code:
            return patient["time"][i] 
    raise ValueError(f"Couldn't find patient birthdate for patient {patient['subject_id'][0]}")


def get_table_indices(patient: datasets.Dataset, table_name: str) -> np.ndarray:
    """
    Get the indices of the rows in the dataset that correspond to a given table name.
    """
    return np.where(np.array(patient["table_name"]) == table_name)[0]

def check_vocab_overlap(model_name: str,
                        patient: datasets.Dataset,
                        code_col: str = 'code'):
    """
    Check if the codes in the patient data are present in the model vocabulary.

    Args:
        model_name: The name of the model to check the vocabulary against.
        patient: The patient dataset to check the vocabulary against.
        code_col: The column name of the codes in the patient data.

    Returns:
        The overlap between the model vocabulary and the patient data.

    Example:
        >>> check_vocab_overlap(model_name="StanfordShahLab/clmbr-t-base",
                                patient=MEDSV3_DATA,
                                code_col="code")
    """
    model_codes = get_model_vocab(model_name=model_name, 
                                  codes_only=True)

    # First, we should check that at least some of the codes in the model are present in the MEDSV3_DATA
    overlap = set(model_codes).intersection(set(patient[code_col]))
    if len(overlap) == 0:
        raise ValueError(f"No overlap between model vocabulary and patient data for model {model_name}")
    else:
        print(f"Overlap between model vocabulary and patient data for model {model_name}: {len(overlap)}")
    return overlap

def filter_dataset(dataset: datasets.Dataset,
                   filter_dict: dict):
    """
    Filter a dataset.

    Args:
        dataset: The dataset to filter.
        filter_dict: A dictionary of column names and values to filter by.

    Returns:
        The filtered dataset.

    Example:
        >>> filter_dataset(dataset=MEDSV3_DATA,
                           filter_dict={"table_name": "measurement",
                                       "code": "Visit/IP"})
    """    
    for key, value in filter_dict.items():
        if not isinstance(value, list):
            value = [value]
        dataset = dataset.filter(lambda x: x[key] in value)
    return dataset

def filter_patients(dataset: datasets.Dataset,
                    subject_ids: list[int] | int | str,
                    subject_id_col: str = 'subject_id'):
    """
    Filter a dataset for a specific set of patient IDs.

    Args:
        dataset: The dataset to filter.
        patient_ids: The list of patient IDs to filter for.
        subject_id_col: The column name of the subject IDs in the dataset.

    Returns:
        The filtered dataset.
    """
    if isinstance(subject_ids, int):
        subject_ids = [subject_ids]
    
    if isinstance(subject_ids, str):
        subject_ids = [int(subject_ids)]

    return filter_dataset(dataset=dataset,
                          filter_dict={subject_id_col: subject_ids})

def get_event_indices(dataset: datasets.Dataset):
    """
    Get the indices of the rows in the dataset that correspond to a given event (i.e. a unique timestamp).

    Args:
        dataset: The dataset to get the event indices from.

    Returns:
        A dictionary of times and their corresponding indices.
    """
    time_indices = {}
    for i, time in enumerate(dataset['time']):
        if time is not None:
            if time not in time_indices:
                time_indices[time] = []
            time_indices[time].append(i)

    return time_indices