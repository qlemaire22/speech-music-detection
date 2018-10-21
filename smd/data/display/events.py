import sed_vis
import dcase_util


def audio_with_events(audio_path, annotation_path, estimation_path=None):
    audio_container = dcase_util.containers.AudioContainer().load(audio_path)
    reference_event_list = dcase_util.containers.MetaDataContainer().load(annotation_path)

    if estimation_path is None:
        event_lists = {'reference': reference_event_list}
    else:
        estimated_event_list = dcase_util.containers.MetaDataContainer().load(estimation_path)
        event_lists = {
            'reference': reference_event_list,
            'estimated': estimated_event_list
        }
    vis = sed_vis.visualization.EventListVisualizer(event_lists=event_lists,
                                                    audio_signal=audio_container.data,
                                                    sampling_rate=audio_container.fs)
    vis.show()
