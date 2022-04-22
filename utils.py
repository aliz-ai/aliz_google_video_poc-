import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.pyplot import figure
import json
from google.cloud import storage
from datetime import datetime
import time


def create_upload_test_json(test_video_urls,
                            project_id = "sharp-leaf-344111",
                            import_file_df = None,
                            test_json_folder = "test_json",
                            bucket_name = "gs://aliz_action_recognition_poc",
                            time_segment_start = 0,
                            time_segment_end = 1000,
                            ):
    
    """
    Given a list of GCS URLS of video to predict, for each video, create a "test json" file needed to perform batch prediction
    and upload the file to a specific folder in a GCS bucket.
    
    Arguments:
        - test_video_urls: a list of GCS of video to predict
        - project_id: gcp project ID
        - import_file_df: a dataframe containing the list of test video URLs and the associated end time stamp. If provided, will be used to extract the 'time_segment_end' (will supersede the time segment end parameter below)
        - time_segment_start: the start of prediction (second),
        - time_segment_end: the prediction won't be performed beyond this timestamp (second),
        - bucket_name: output GCS bucket
        - test_json_folder: a folder in the GCS bucket into which the test jsons will be created.
    
    Returns:
        - test_json_urls: list of GCS URLs of the test json files
    """
    test_json_urls = []
    for idx, test_video_url in enumerate(test_video_urls):   
        # Grab the ending time segment for each testing video from the `import_file_df` dataframe
        test_video_name = test_video_url.split("/")[-1]
        
        if import_file_df is not None:
            time_segment_end = import_file_df.loc[import_file_df.gcs_url == test_video_url, "end"].values[0]
            
        # Configure the test-data, first as a dictionary, and then dump it as a json file
        data_dict = {
            "content": test_video_url,
            "mimeType": "video/avi",
            "timeSegmentStart": f"{time_segment_start}s",
            "timeSegmentEnd": f"{time_segment_end}s",
        }

        json_file = f"pred_{test_video_name}.json"
        data_str = json.dumps(data_dict) + "\n" 

        # Upload the json to GCS bucket
        bucket = storage.Client(project=project_id).bucket(bucket_name.replace("gs://", ""))
        blob = bucket.blob(blob_name=f"{test_json_folder}/" +json_file)
        blob.upload_from_string(data_str)

        # Check if the json is uploaded to GCS successfully
        test_json_url = f"{bucket_name}/{test_json_folder}/" + json_file
        print(f"{idx + 1}. The video '{test_video_name}' has been inputted into the test json '{test_json_url}'")
        test_json_urls.append(test_json_url)
        print("="*100)
        
    return test_json_urls
    

def do_batch_prediction(test_json_urls,
                        model,
                        bucket_name = "gs://aliz_action_recognition_poc"):
    """
    Given a list of test_json_urls and a trained model object, perform a batch prediction and output the result on the given bucket
    """
    
    PRED_TIMESTAMP = datetime.now().strftime("%Y/%m/%d_%H:%M:%S")
    
    for idx, test_json_url in enumerate(test_json_urls):
        video_name = test_json_url.split("/")[-1][5:-5]
        print("="*100)
        print(f"Creating prediction job for video '{video_name}':")
        batch_predict_job = model.batch_predict(
            job_display_name= PRED_TIMESTAMP+ f"__{video_name}", # each video will get one predict job
            gcs_source=test_json_url,
            gcs_destination_prefix=bucket_name,
            sync=False,
        )
        print(batch_predict_job)
        time.sleep(5) # so that the jobs don't get submitted concurrently which will lead to failure

        
        
def combine_true_pred_label(true_label_df, 
                            pred_label_df, 
                            eps = 2, 
                            true_label_col = "timestamp", 
                            pred_label_col = "timeSegmentEnd",
                            max_time = None,
                            min_confidence = 0.998):
    """
    Combines true_label_df and true_label_df, aligning each prediction with the true timestamp that is within 'eps' distance from it.
    Args:
        - true_label_df: a dataframe containing the "true" timestamps for a given video
        - pred_label_df: a dataframe containing predicted timestamps for a given video
        - eps: epsilon, the maximum difference between a true label and a predicted timestamp for this pair to be considered matching
        - true_label_col: the name of column in true_label_df that stores the true labels
        - pred_label_col: the name of column in pred_label_df that stores the predicted labels
        - max_time: the maximum time beyond which all the predictions and true labels will be discarded
        - min_confidence: the min_confidence for a prediction to be considered a valid prediction. Usually true positives have very high confidence (> 0.999) so setting a high number for min_confidence is encouraged.
    Returns:
        - result_df: a dataframe containing the predictions that have been aligned with true labels
    """
    
    true_label_df = true_label_df.copy()
    true_label_df["prediction"] = None # create a new column for prediction
    
    for row in true_label_df.iterrows():
        index = row[0]
        true_timestamp = row[1][true_label_col]
        pred_timestamps = pred_label_df[pred_label_col]
        matching_pred = pred_label_df[abs(true_timestamp - pred_timestamps) <= eps][pred_label_col]
        if len(matching_pred) == 0:
            matching_pred = None
        else:
            if len(matching_pred) > 1:
                print(f'Warning: more than two predictions ({len(matching_pred)}) can be matched to the true timestamp at {true_timestamp}, consider decreasing the epsilon.')
            true_label_df.at[index, "prediction"] = matching_pred.values[0]
            
    true_label_df["prediction"] =  true_label_df["prediction"].astype(float)
    result_df = pd.merge(true_label_df, pred_label_df, left_on='prediction', right_on=pred_label_col, how = "outer")
    result_df = result_df[[true_label_col, pred_label_col, "confidence"]]
    result_df.columns = ["true_label", "pred", "confidence"]
    
    def _get_average(true_label, pred):
        # get the average between true_label and pred
        if pd.notnull(true_label) and pd.notnull(pred):
            return (true_label + pred)/2
        elif pd.notnull(true_label):
            return true_label
        else:
            return pred
        
    result_df["ave"] = result_df.apply(lambda x: _get_average(x["true_label"], x["pred"]), axis = 1)
    result_df = result_df.sort_values(by = ["ave"], ascending = True)
    if max_time:
        result_df = result_df[result_df["ave"] <= max_time]
    result_df = result_df.drop("ave", axis = 1)
    
    
    def _put_label(true_label, pred):
        # compare true_label and pred and classify the comparison outcome into one of "true positive", "true negative" or "false positive"
        if pd.notnull(true_label) and pd.notnull(pred):
            return "true positive"
        elif pd.notnull(true_label):
            return"false negative"
        else:
            return "false positive"
        
    result_df["label"] = result_df.apply(lambda x: _put_label(x['true_label'], x['pred']), axis=1)
    result_df = result_df.reset_index(drop = True)
    
    if min_confidence:
        result_df = result_df.loc[(result_df["confidence"] >= min_confidence) | (result_df["confidence"].isnull())]

    return result_df



def pretty_display(result_df):
    """
    Apply conditional formatting to result_df and display it
    """
    def _highlight_cells(val):
        if val == "true positive":
            color = 'palegreen' 
        elif val == "false positive":
            color = "yellow"
        elif val == "false negative":
            color = "pink"
        else:
            color = ''

        return 'background-color: {}'.format(color)

    display(result_df.style.applymap(lambda x: 'color: red' if pd.isnull(x) else '').applymap(_highlight_cells))
        

def get_classification_metrics(result_df):
    """
    Given a result_df, get the usual classification metrics out of it
    """
    real_positive = result_df.notna().sum()["true_label"]
    pred_positive = result_df.notna().sum()["pred"]
    true_positive = len(result_df[result_df.label == "true positive"])
    false_positive = len(result_df[result_df.label == "false positive"])
    false_negative = len(result_df[result_df.label == "false negative"])
    recall = true_positive/real_positive
    precision = true_positive/pred_positive
    
    return {"real_positive": real_positive, "pred_positive": pred_positive, "true_positive": true_positive, "false_positive": false_positive, "false_negative": false_negative, "recall": recall, "precision": precision}




def draw_piechart(result_df):
    """
    Given a result_df, draw a pie chart depicting the ratio of true positive, false positive and false negative
    """
    true_positive = len(result_df[result_df.label == "true positive"])
    false_positive = len(result_df[result_df.label == "false positive"])
    false_negative = len(result_df[result_df.label == "false negative"])
    labels = 'FP', 'FN', 'TP'
    sizes = [false_positive, false_negative, true_positive]
    explode = (0, 0, 0.1)  # only "explode" the 2nd slice (i.e. 'Hogs')
    fig1, ax1 = plt.subplots(figsize=(10, 10), dpi=80)
    ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90, colors = ["yellow","pink","lightgreen"])
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.show()
    
