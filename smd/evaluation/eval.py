import sed_eval
import dcase_util
from tqdm import tqdm


def eval(ground_truth_events, predicted_events, segment_length=0.01, event_tolerance=0.2, offset=False):
    r"""
        Evaluate the output of the network.

        ground_truth_events and predicted_events can either be:
            - a list of path to containing the events (one path for one audio)
            - a path containing the events (for one audio)
            - a list of list containing the events (a list of the events of different audio)
            - a list containing the events (the events of one audio)

        The segment_length and the event_tolerance are giving in s.
        If one of those two parameters is set to None, this evaluation will be skipped.
    """
    data = []
    all_data = dcase_util.containers.MetaDataContainer()

    if type(ground_truth_events) == str and type(predicted_events) == str:
        reference_event_list = sed_eval.io.load_event_list(ground_truth_events)
        estimated_event_list = sed_eval.io.load_event_list(predicted_events)

        data.append({'reference_event_list': reference_event_list,
                     'estimated_event_list': estimated_event_list})

        all_data += reference_event_list

    elif type(ground_truth_events) == list and type(predicted_events) == list:
        if len(ground_truth_events) != len(predicted_events):
            raise ValueError("The two lists must have the same size.")
        if all(isinstance(n, str) for n in ground_truth_events) and all(isinstance(n, str) for n in predicted_events):

            for i in range(len(ground_truth_events)):
                reference_event_list = sed_eval.io.load_event_list(ground_truth_events[i])
                estimated_event_list = sed_eval.io.load_event_list(predicted_events[i])

                data.append({'reference_event_list': reference_event_list,
                             'estimated_event_list': estimated_event_list})

                all_data += reference_event_list

        if all(isinstance(n, list) for n in ground_truth_events) and all(isinstance(n, list) for n in predicted_events):
            if all(isinstance(x, list) for b in ground_truth_events for x in b) and all(isinstance(x, list) for b in predicted_events for x in b):
                for gt, p in zip(ground_truth_events, predicted_events):
                    formatted_gt_events = []
                    formatted_p_events = []

                    for event in gt:
                        formatted_gt_events.append({'onset': event[0], 'offset': event[1], 'event_label': event[2]})

                    for event in p:
                        formatted_p_events.append({'onset': event[0], 'offset': event[1], 'event_label': event[2]})

                    formatted_p_events = dcase_util.containers.MetaDataContainer(formatted_p_events)
                    formatted_gt_events = dcase_util.containers.MetaDataContainer(formatted_gt_events)

                    data.append({'reference_event_list': formatted_gt_events,
                                 'estimated_event_list': formatted_p_events})

                    all_data += formatted_gt_events
            else:
                formatted_gt_events = []
                formatted_p_events = []

                for event in ground_truth_events:
                    formatted_gt_events.append({'onset': event[0], 'offset': event[1], 'event_label': event[2]})

                for event in predicted_events:
                    formatted_p_events.append({'onset': event[0], 'offset': event[1], 'event_label': event[2]})

                formatted_p_events = dcase_util.containers.MetaDataContainer(formatted_p_events)
                formatted_gt_events = dcase_util.containers.MetaDataContainer(formatted_gt_events)

                data.append({'reference_event_list': formatted_gt_events,
                             'estimated_event_list': formatted_p_events})

                all_data += reference_event_list
    else:
        raise ValueError("Incorrect input format.")

    event_labels = all_data.unique_event_labels

    if not(segment_length is None):
        segment_based_metrics = sed_eval.sound_event.SegmentBasedMetrics(
            event_label_list=event_labels,
            time_resolution=segment_length
        )

    if not(event_tolerance is None):
        event_based_metrics = sed_eval.sound_event.EventBasedMetrics(
            event_label_list=event_labels,
            t_collar=event_tolerance,
            percentage_of_length=0.,
            evaluate_onset=True,
            evaluate_offset=offset
        )

    for file_pair in tqdm(data):
        if not(segment_length is None):
            segment_based_metrics.evaluate(
                reference_event_list=file_pair['reference_event_list'],
                estimated_event_list=file_pair['estimated_event_list']
            )
        if not(event_tolerance is None):
            event_based_metrics.evaluate(
                reference_event_list=file_pair['reference_event_list'],
                estimated_event_list=file_pair['estimated_event_list']
            )

    if not(event_tolerance is None) and not(segment_length is None):
        print(segment_based_metrics)
        print(event_based_metrics)
        return segment_based_metrics, event_based_metrics
    elif event_tolerance is None and not(segment_length is None):
        print(segment_based_metrics)
        return segment_based_metrics
    elif not(event_tolerance is None) and segment_length is None:
        print(event_based_metrics)
        return event_based_metrics
    else:
        return
