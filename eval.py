import sed_eval
import dcase_util

file_list = [
    {
        'reference_file': 'office_snr0_high_v2.txt',
        'estimated_file': 'office_snr0_high_v2_detected.txt'
    },
    {
        'reference_file': 'office_snr0_med_v2.txt',
        'estimated_file': 'office_snr0_med_v2_detected.txt'
    }
]

data = []

# Get used event labels
all_data = dcase_util.containers.MetaDataContainer()
for file_pair in file_list:
    reference_event_list = sed_eval.io.load_event_list(
        file_pair['reference_file']
    )
    estimated_event_list = sed_eval.io.load_event_list(
        filename=file_pair['estimated_file']
    )

    data.append({'reference_event_list': reference_event_list,
                 'estimated_event_list': estimated_event_list})

    all_data += reference_event_list

event_labels = all_data.unique_event_labels

# Start evaluating

# Create metrics classes, define parameters
segment_based_metrics = sed_eval.sound_event.SegmentBasedMetrics(
    event_label_list=event_labels,
    time_resolution=1.0
)

event_based_metrics = sed_eval.sound_event.EventBasedMetrics(
    event_label_list=event_labels,
    t_collar=0.250
)

# Go through files
for file_pair in data:
    segment_based_metrics.evaluate(
        reference_event_list=file_pair['reference_event_list'],
        estimated_event_list=file_pair['estimated_event_list']
    )

    event_based_metrics.evaluate(
        reference_event_list=file_pair['reference_event_list'],
        estimated_event_list=file_pair['estimated_event_list']
    )

# Get only certain metrics
overall_segment_based_metrics = segment_based_metrics.results_overall_metrics()
print("Accuracy:", overall_segment_based_metrics['accuracy']['accuracy'])

# Or print all metrics as reports
print(segment_based_metrics)
print(event_based_metrics)
