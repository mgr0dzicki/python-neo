# -*- coding: utf-8 -*-

"""
Module for reading and writing NSDF files

Author: Mieszko Grodzicki

This module support both reading and writing NDSF files.
Note: Read file must be written using this IO
"""

from __future__ import absolute_import

import numpy as np
import quantities as pq

from uuid import uuid1
import pickle
from datetime import datetime
import os

try:
    import nsdf
    import h5py
except ImportError as err:
    HAVE_NSDF = False
    NSDF_ERR = err
else:
    HAVE_NSDF = True
    NSDF_ERR = None

from neo.io.baseio import BaseIO
from neo.core import Block, Segment, AnalogSignal, IrregularlySampledSignal, \
    Event, ChannelIndex


class NSDFIO(BaseIO):
    """
    Class for reading and writing files in NSDF Format.

    It supports reading and writing: Block, Segment, AnalogSignal, IrregularlySampledSignal,
    Event, ChannelIndex, with all relationships and metadata.
    """
    is_readable = True
    is_writable = True

    supported_objects = [Block, Segment, ChannelIndex, AnalogSignal,
                         IrregularlySampledSignal, Event]

    readable_objects = [Block, Segment]
    writeable_objects = [Block, Segment]

    has_header = False
    is_streameable = False

    name = 'NSDF'
    extensions = ['h5']
    mode = 'file'

    def __init__(self, filename=None):
        """
        Initialise NSDFIO instance

        :param filename: Path to the file
        """
        if not HAVE_NSDF:
            raise Exception("Failed to import NSDF.")

        if filename is None:
            raise ValueError("Must provide an input file.")

        BaseIO.__init__(self)

        self.filename = filename
        self.dt_format = '%d/%m/%Y %H:%M:%S'
        self.modeltree_path = '/model/modeltree/neo/'

    def write_all_blocks(self, blocks):
        """
        Write list of blocks to the file

        :param blocks: List of blocks to be written
        """
        writer = self._init_writing()
        neo_model, blocks_model, segments_model = self._prepare_model_tree(writer)

        name_pattern = self._name_pattern(len(blocks))
        for i, block in enumerate(blocks):
            self.write_block(block, name_pattern.format(i), blocks_model, writer)

    def write_block(self, block=None, name='0', parent=None, writer=None):
        """
        Write a Block to the file

        :param block: Block to be written
        :param name: Name for block representation in NSDF model tree (optional)
        :param writer: NSDFWriter instance (optional)
        :param parent: NSDF ModelComponent which will be the parent of block NSDF representation (optional)
        """
        if not isinstance(block, Block):
            raise ValueError("Must provide a Block to write.")

        if writer is None:
            writer = self._init_writing()

        if parent is None:
            neo_model, parent, segments_model = self._prepare_model_tree(writer)

        block_model = nsdf.ModelComponent(name, uid=uuid1().hex, parent=parent)
        self._write_container_metadata(block, block_model)
        self._write_model_component(block_model, writer)

        self._write_block_children(block, block_model, writer)

        self._clean_nsdfio_annotations(block)

    def _write_block_children(self, block, block_model, writer):
        self._write_children(block.segments, 'segments', 'write_segment', block_model, writer)
        self._write_children(block.channel_indexes, 'channel_indexes', 'write_channelindex',
                             block_model, writer)

    def write_segment(self, segment=None, name='0', writer=None, parent=None):
        """
        Write a Segment to the file

        :param segment: Segment to be written
        :param name: Name for segment representation in NSDF model tree (optional)
        :param writer: NSDFWriter instance (optional)
        :param parent: NSDF ModelComponent which will be the parent of segment NSDF representation (optional)
        """
        if not isinstance(segment, Segment):
            raise ValueError("Must provide a Segment to write.")

        if writer is None:
            writer = self._init_writing()

        single_segment = False
        if parent is None:
            neo_model, blocks_model, parent = self._prepare_model_tree(writer)
            single_segment = True

        model = nsdf.ModelComponent(name, uid=uuid1().hex, parent=parent)
        self._write_container_metadata(segment, model)
        self._write_model_component(model, writer)

        self._write_segment_children(model, segment, writer)

        if single_segment:
            self._clean_nsdfio_annotations(segment)

    def _write_segment_children(self, model, segment, writer):
        self._write_children(segment.analogsignals, 'analogsignals', 'write_analogsignal', model, writer)
        self._write_children(segment.irregularlysampledsignals, 'irregularlysampledsignals',
                             'write_irregularlysampledsignal', model, writer)
        self._write_children(segment.events, 'events', 'write_event', model, writer)

    def write_analogsignal(self, signal, name, writer, parent):
        """
        Write an AnalogSignal to the file

        :param signal: AnalogSignal to be written
        :param name: Name for signal representation in NSDF model tree
        :param writer: NSDFWriter instance
        :param parent: NSDF ModelComponent which will be the parent of signal NSDF representation
        """
        self._write_signal(signal, name, writer, parent)

    def write_irregularlysampledsignal(self, signal, name, writer, parent):
        """
        Write an IrregularlySampledSignal to the file

        :param signal: IrregularlySampledSignal to be written
        :param name: Name for signal representation in NSDF model tree
        :param writer: NSDFWriter instance
        :param parent: NSDF ModelComponent which will be the parent of signal NSDF representation
        """
        self._write_signal(signal, name, writer, parent)

    def _write_signal(self, signal, name, writer, parent):
        uid = uuid1().hex
        model = nsdf.ModelComponent(name, uid=uid, parent=parent)
        regular = isinstance(signal, AnalogSignal)

        if self._write_only_reference(model, signal, uid, writer):
            return

        self._write_basic_metadata(model, signal)

        r_signal = np.swapaxes(signal, 0, 1)
        channels_model, channels, source_ds = self._create_signal_data_sources(model, len(r_signal),
                                                                               uid, writer, regular)
        self._write_signal_data(model, channels, r_signal, signal, source_ds, writer)

        self._write_model_tree(model, writer)
        signal.annotations['nsdfio_uid'] = uid

    def write_event(self, event, name, writer, parent):
        """
        Write an Event to the file

        :param signal: Event to be written
        :param name: Name for event representation in NSDF model tree
        :param writer: NSDFWriter instance
        :param parent: NSDF ModelComponent which will be the parent of event NSDF representation
        """
        uid = uuid1().hex
        model = nsdf.ModelComponent(name, uid=uid, parent=parent)

        if self._write_only_reference(model, event, uid, writer):
            return

        self._write_basic_metadata(model, event)
        self._write_model_component(model, writer)

        source_ds, source_name_dict = self._create_event_data_sources(model, uid, writer)
        self._write_event_data(event, model, source_ds, source_name_dict, writer)

        event.annotations['nsdfio_uid'] = uid

    def write_channelindex(self, channelindex, name, writer, parent):
        """
        Write a ChannelIndex to the file

        :param channelindex: ChannelIndex to be written
        :param name: Name for channelindex representation in NSDF model tree
        :param writer: NSDFWriter instance
        :param parent: NSDF ModelComponent which will be the parent of channelindex NSDF representation
        """
        uid = uuid1().hex
        model = nsdf.ModelComponent(name, uid=uid, parent=parent)

        self._write_basic_metadata(model, channelindex)
        self._write_model_component(model, writer)

        self._write_channelindex_arrays(model, channelindex, writer)

        self._write_channelindex_children(channelindex, model, writer)

    def _write_channelindex_children(self, channelindex, model, writer):
        self._write_children(channelindex.analogsignals, 'analogsignals', 'write_analogsignal', model, writer)

    def _write_children(self, children, name, function, parent_model, writer):
        model = nsdf.ModelComponent(name=name, uid=uuid1().hex, parent=parent_model)
        self._write_model_component(model, writer)
        name_pattern = self._name_pattern(len(children))
        for i, child in enumerate(children):
            getattr(self, function)(child, name_pattern.format(i), writer, model)

    def _init_writing(self):
        return nsdf.NSDFWriter(self.filename, mode='w', dialect=nsdf.dialect.NUREGULAR)

    def _prepare_model_tree(self, writer):
        neo_model = nsdf.ModelComponent('neo', uid=uuid1().hex)
        self._write_model_component(neo_model, writer)

        blocks_model = nsdf.ModelComponent('blocks', uid=uuid1().hex, parent=neo_model)
        self._write_model_component(blocks_model, writer)

        segments_model = nsdf.ModelComponent('segments', uid=uuid1().hex, parent=neo_model)
        self._write_model_component(segments_model, writer)

        return neo_model, blocks_model, segments_model

    def _number_of_digits(self, n):
        return len(str(n))

    def _name_pattern(self, how_many_items):
        return '{{:0{}d}}'.format(self._number_of_digits(max(how_many_items - 1, 0)))

    def _clean_nsdfio_annotations(self, object):
        nsdfio_annotations = ('nsdfio_uid',)

        for key in nsdfio_annotations:
            object.annotations.pop(key, None)

        if hasattr(object, 'children'):
            for child in object.children:
                self._clean_nsdfio_annotations(child)

    def _write_only_reference(self, model, object, uid, writer):
        if object.annotations.get('nsdfio_uid') is not None:
            model.attrs['reference_to'] = object.annotations['nsdfio_uid']
            self._write_model_component(model, writer)
            return True
        return False

    def _write_model_component(self, model, writer):
        if model.parent is None:
            nsdf.add_model_component(model, writer.model['modeltree/'])
        else:
            nsdf.add_model_component(model, model.parent.hdfgroup)

    def _write_model_tree(self, model, writer):
        self._write_model_component(model, writer)
        for child in model.children.values():
            self._write_model_tree(child, writer)

    def _write_container_metadata(self, container, container_model):
        self._write_basic_metadata(container_model, container)

        self._write_datetime_attributes(container_model, container)
        self._write_index_attribute(container_model, container)

    def _write_basic_metadata(self, model, object):
        self._write_basic_attributes(model, object)
        self._write_annotations(model, object)

    def _write_basic_attributes(self, model, object):
        if object.name is not None:
            model.attrs['name'] = self._encode_string(object.name)
        if object.description is not None:
            model.attrs['description'] = self._encode_string(object.description)

    def _write_datetime_attributes(self, model, object):
        if object.rec_datetime is not None:
            model.attrs['rec_datetime'] = object.rec_datetime.strftime(self.dt_format)

    def _write_index_attribute(self, model, object):
        if object.index is not None:
            model.attrs['index'] = object.index

    def _write_annotations(self, model, object):
        if object.annotations is not None:
            object.annotations.pop('nsdfio_path', None)
            model.attrs['annotations'] = self._encode_string(pickle.dumps(object.annotations, 0))

    def _write_signal_data(self, model, channels, r_signal, signal, source_ds, writer):
        regular = isinstance(signal, AnalogSignal)

        if regular:
            self._write_analogsignal_data(model, channels, r_signal, signal, source_ds, writer)
        else:
            self._write_irregularlysampledsignal_data(model, channels, r_signal, signal, source_ds, writer)

    def _write_analogsignal_data(self, model, channels, r_signal, signal, source_ds, writer):
        dataobj = nsdf.UniformData('signal', unit=str(signal.units.dimensionality))
        dataobj.set_dt(float(signal.sampling_period.magnitude),
                       str(signal.sampling_period.dimensionality))
        dataobj.dtype = signal.dtype

        for i in range(len(channels)):
            dataobj.put_data(channels[i].uid, r_signal[i])

        rescaled_tstart = signal.t_start.rescale(signal.sampling_period.dimensionality)
        writer.add_uniform_data(source_ds, dataobj,
                                tstart=float(rescaled_tstart.magnitude))

        model.attrs['t_start_unit'] = str(signal.t_start.dimensionality)

    def _write_irregularlysampledsignal_data(self, model, channels, r_signal, signal, source_ds, writer):
        dataobj = nsdf.NonuniformRegularData('signal', unit=str(signal.units.dimensionality))
        dataobj.set_times(signal.times, str(signal.times.units.dimensionality))
        dataobj.dtype = signal.dtype

        for i in range(len(channels)):
            dataobj.put_data(channels[i].uid, r_signal[i])

        writer.add_nonuniform_regular(source_ds, dataobj)

    def _create_signal_data_sources(self, model, channels_number, uid, writer, regular):
        channels = []
        channels_model = nsdf.ModelComponent(name='channels', uid=uuid1().hex, parent=model)
        name_pattern = '{{:0{}d}}'.format(self._number_of_digits(max(channels_number - 1, 0)))
        for i in range(channels_number):
            channels.append(nsdf.ModelComponent(name_pattern.format(i),
                                                uid=uuid1().hex,
                                                parent=channels_model))

        if regular:
            source_ds = writer.add_uniform_ds(uid, [channel.uid.encode() for channel in channels])
        else:
            source_ds = writer.add_nonuniform_ds(uid, [channel.uid.encode() for channel in channels])
        return channels_model, channels, source_ds

    def _write_event_data(self, event, model, source_ds, source_name_dict, writer):
        dataobj = nsdf.EventData('times', unit=str(event.units.dimensionality))
        dataobj.put_data(model.uid, event.times)
        writer.add_event_1d(source_ds, dataobj, source_name_dict)
        self._write_array(model.hdfgroup, 'labels', event.labels)

    def _create_event_data_sources(self, model, uid, writer):
        source_ds = writer.add_event_ds_1d(uid, 'times', [uid])
        source_name_dict = {}
        source_name_dict[uid] = 'data'
        return source_ds, source_name_dict

    def _write_channelindex_arrays(self, model, channelindex, writer):
        group = model.hdfgroup

        self._write_array(group, 'index', channelindex.index)
        if channelindex.channel_names is not None:
            self._write_array(group, 'channel_names', channelindex.channel_names)
        if channelindex.channel_ids is not None:
            self._write_array(group, 'channel_ids', channelindex.channel_ids)
        if channelindex.coordinates is not None:
            self._write_array(group, 'coordinates', channelindex.coordinates)

    def _write_array(self, group, name, array):
        if isinstance(array, pq.Quantity):
            group.create_dataset(name, data=array.magnitude)
            group[name].attrs['dimensionality'] = str(array.dimensionality)
        elif isinstance(array, np.ndarray):
            if array.dtype.type == np.str_:
                array = np.void(np.char.encode(array))
            elif array.dtype.type == np.bytes_:
                array = np.void(array)
            group.create_dataset(name, data=array)
        else:
            group.create_dataset(name, data=array)
            group[name].attrs['is_list'] = 'True'

    def _encode_string(self, string):
        if isinstance(string, str):
            string = string.encode()
        return np.void(string)

    def read_all_blocks(self, lazy=False, cascade=True):
        """
        Read all blocks from the file

        :param lazy: Enables lazy reading
        :param cascade: Read nested objects or not?
        :return: List of read blocks
        """
        reader = self._init_reading()
        blocks = []

        blocks_path = self.modeltree_path + 'blocks/'
        for block in reader.model[blocks_path].values():
            blocks.append(self.read_block(lazy, cascade, group=block, reader=reader))

        return blocks

    def read_block(self, lazy=False, cascade=True, group=None, reader=None):
        """
        Read a Block from the file

        :param lazy: Enables lazy reading
        :param cascade: Read nested objects or not?
        :param group: HDF5 Group representing the block in NSDF model tree (optional)
        :param reader: NSDFReader instance (optional)
        :return: Read block
        """
        block = Block()
        group, reader, single_block = self._select_first_container(group, reader, 'block')

        if group is None:
            return None

        attrs = group.attrs

        if cascade:
            self._read_block_children(lazy, block, group, reader)
        block.create_many_to_one_relationship()

        self._read_container_metadata(attrs, block, path=group.name)

        return block

    def _read_block_children(self, lazy, block, group, reader):
        for child in group['segments/'].values():
            block.segments.append(self.read_segment(lazy=lazy, group=child, reader=reader))
        for child in group['channel_indexes/'].values():
            block.channel_indexes.append(self.read_channelindex(lazy=lazy, group=child, reader=reader))

    def read_segment(self, lazy=False, cascade=True, group=None, reader=None):
        """
        Read a Segment from the file

        :param lazy: Enables lazy reading
        :param cascade: Read nested objects or not?
        :param group: HDF5 Group representing the segment in NSDF model tree (optional)
        :param reader: NSDFReader instance (optional)
        :return: Read segment
        """
        segment = Segment()
        group, reader, single_segment = self._select_first_container(group, reader, 'segment')

        if group is None:
            return None

        attrs = group.attrs

        if cascade:
            self._read_segment_children(lazy, group, reader, segment)
        if single_segment:
            segment.create_many_to_one_relationship()

        self._read_container_metadata(attrs, segment, path=group.name)

        return segment

    def _read_segment_children(self, lazy, group, reader, segment):
        for child in group['analogsignals/'].values():
            segment.analogsignals.append(self.read_analogsignal(lazy=lazy, group=child, reader=reader))
        for child in group['irregularlysampledsignals/'].values():
            segment.irregularlysampledsignals.append(self.read_irregularlysampledsignal(lazy=lazy, group=child,
                                                                                        reader=reader))
        for child in group['events/'].values():
            segment.events.append(self.read_event(lazy=lazy, group=child, reader=reader))

    def read_analogsignal(self, lazy=False, cascade=True, group=None, reader=None):
        """
        Read an AnalogSignal from the file (must be child of a Segment)

        :param lazy: Enables lazy reading
        :param cascade: Read nested objects or not?
        :param group: HDF5 Group representing the analogsignal in NSDF model tree
        :param reader: NSDFReader instance
        :return: Read AnalogSignal
        """
        attrs = group.attrs

        if attrs.get('reference_to') is not None:
            return self.objects_dict[attrs['reference_to']]

        uid = attrs['uid']
        data_group = reader.data['uniform/{}/signal'.format(uid)]

        t_start = self._read_analogsignal_t_start(attrs, data_group)
        signal = self._create_analogsignal(data_group, lazy, group, t_start, uid, reader)

        self._read_basic_metadata(attrs, signal, path=group.name)

        self.objects_dict[uid] = signal
        return signal

    def read_irregularlysampledsignal(self, lazy=False, cascade=True, group=None, reader=None):
        """
        Read an IrregularlySampledSignal from the file (must be child of a Segment)

        :param lazy: Enables lazy reading
        :param cascade: Read nested objects or not?
        :param group: HDF5 Group representing the irregularlysampledsignal in NSDF model tree
        :param reader: NSDFReader instance
        :return: Read AnalogSignal
        """
        attrs = group.attrs

        if attrs.get('reference_to') is not None:
            return self.objects_dict[attrs['reference_to']]

        uid = attrs['uid']
        data_group = reader.data['nonuniform/{}/signal'.format(uid)]
        signal = self._create_irregularlysampledsignal(data_group, lazy, group, uid, reader)

        self._read_basic_metadata(attrs, signal, path=group.name)

        self.objects_dict[uid] = signal
        return signal

    def read_event(self, lazy=False, cascade=True, group=None, reader=None):
        """
        Read an Event from the file (must be child of a Segment)

        :param lazy: Enables lazy reading
        :param cascade: Read nested objects or not?
        :param group: HDF5 Group representing the event in NSDF model tree
        :param reader: NSDFReader instance
        :return: Read Event
        """
        attrs = group.attrs

        if attrs.get('reference_to') is not None:
            return self.objects_dict[attrs['reference_to']]

        uid = attrs['uid']
        data_group = reader.data['event/{}/times/data'.format(uid)]
        event = self._create_event(data_group, group, lazy, reader, uid)

        self._read_basic_metadata(attrs, event, path=group.name)

        self.objects_dict[uid] = event
        return event

    def read_channelindex(self, lazy=False, cascade=True, group=None, reader=None):
        """
        Read a ChannelIndex from the file (must be child of a Block)

        :param lazy: Enables lazy reading
        :param cascade: Read nested objects or not?
        :param group: HDF5 Group representing the channelindex in NSDF model tree
        :param reader: NSDFReader instance
        :return: Read ChannelIndex
        """
        attrs = group.attrs

        channelindex = self._create_channelindex(group)
        if cascade:
            self._read_channelindex_children(lazy, group, reader, channelindex)

        self._read_basic_metadata(attrs, channelindex, path=group.name)

        return channelindex

    def _read_channelindex_children(self, lazy, group, reader, channelindex):
        for child in group['analogsignals/'].values():
            channelindex.analogsignals.append(self.read_analogsignal(lazy=lazy, group=child, reader=reader))

    def load_lazy_object(self, object):
        type = object.__class__.__name__.lower()
        function = getattr(self, 'read_' + type);
        reader = self._init_reading()
        group = reader.model[object.annotations['nsdfio_path']]

        return function(lazy=False, cascade=False, group=group, reader=reader)

    def _init_reading(self):
        reader = nsdf.NSDFReader(self.filename)
        self.file_datetime = datetime.fromtimestamp(os.stat(self.filename).st_mtime)
        self.objects_dict = {}
        return reader

    def _select_first_container(self, group, reader, name):
        if reader is None:
            reader = self._init_reading()

        single = False
        if group is None:
            path = self.modeltree_path + name + 's/'
            if len(reader.model[path].values()) > 0:
                group = list(reader.model[path].values())[0]
            single = True

        return group, reader, single

    def _read_container_metadata(self, attrs, container, path):
        self._read_basic_metadata(attrs, container, path)

        self._read_datetime_attributes(attrs, container)
        self._read_index_attribute(attrs, container)

    def _read_basic_metadata(self, attrs, signal, path):
        self._read_basic_attributes(attrs, signal)
        self._read_annotations(attrs, signal)
        signal.annotations['nsdfio_path'] = path

    def _read_basic_attributes(self, attrs, object):
        if attrs.get('name') is not None:
            object.name = self._decode_string(attrs['name'])
        if attrs.get('description') is not None:
            object.description = self._decode_string(attrs['description'])
        object.file_origin = self.filename

    def _read_datetime_attributes(self, attrs, object):
        object.file_datetime = self.file_datetime
        if attrs.get('rec_datetime') is not None:
            object.rec_datetime = datetime.strptime(attrs['rec_datetime'], self.dt_format)

    def _read_annotations(self, attrs, object):
        if attrs.get('annotations') is not None:
            object.annotations = pickle.loads(attrs['annotations'])

    def _read_index_attribute(self, attrs, object):
        if attrs.get('index') is not None:
            object.index = int(attrs['index'])

    def _create_analogsignal(self, data_group, lazy, group, t_start, uid, reader):
        if lazy:
            data_shape = data_group.shape
            data_shape = (data_shape[1], data_shape[0])
            signal = self._create_lazy_analogsignal(data_shape, data_group, uid, t_start)
        else:
            dataobj = reader.get_uniform_data(uid, 'signal')
            data = self._read_signal_data(dataobj, group)
            signal = self._create_normal_analogsignal(data, dataobj, uid, t_start)
        return signal

    def _create_irregularlysampledsignal(self, data_group, lazy, group, uid, reader):
        if lazy:
            data_shape = (data_group.shape[1], data_group.shape[0])
            data = []
            times = []
        else:
            dataobj = reader.get_nonuniform_data(uid, 'signal')
            data = np.swapaxes(self._read_signal_data(dataobj, group), 0, 1);
            times = dataobj.get_times()

        signal = IrregularlySampledSignal(times, data, units=data_group.attrs['unit'], dtype=data_group.dtype,
                                          time_units=reader.mapping['time/{}_signal'.format(uid)].attrs['unit'])

        if lazy:
            signal.lazy_shape = data_shape

        return signal

    def _create_event(self, data_group, group, lazy, reader, uid):
        if lazy:
            times = []
            labels = np.array([], dtype='S')
        else:
            dataobj = reader.get_event_data(uid, 'times')
            times = dataobj.get_data(uid)
            labels = self._read_array(group, 'labels')
        event = Event(times=times, units=data_group.attrs['unit'], labels=labels)
        if lazy:
            event.lazy_shape = (data_group.shape[0],)
        return event

    def _read_analogsignal_t_start(self, attrs, data_group):
        t_start = float(data_group.attrs['tstart']) * pq.Quantity(1, data_group.attrs['tunit'])
        t_start = t_start.rescale(attrs['t_start_unit'])
        return t_start

    def _read_signal_data(self, dataobj, group):
        data = []
        for channel in group['channels/'].values():
            channel_uid = channel.attrs['uid']
            data += [dataobj.get_data(channel_uid)]
        return data

    def _create_normal_analogsignal(self, data, dataobj, uid, t_start):
        return AnalogSignal(np.swapaxes(data, 0, 1), dtype=dataobj.dtype, units=dataobj.unit,
                            t_start=t_start, sampling_period=pq.Quantity(dataobj.dt, dataobj.tunit))

    def _create_lazy_analogsignal(self, shape, data, uid, t_start):
        attrs = data.attrs
        signal = AnalogSignal([], dtype=data.dtype, units=attrs['unit'],
                              t_start=t_start, sampling_period=pq.Quantity(attrs['dt'], attrs['tunit']))
        signal.lazy_shape = shape
        return signal

    def _create_channelindex(self, group):
        return ChannelIndex(index=self._read_array(group, 'index'),
                            channel_names=self._read_array(group, 'channel_names'),
                            channel_ids=self._read_array(group, 'channel_ids'),
                            coordinates=self._read_array(group, 'coordinates'))

    def _read_array(self, group, name):
        if group.__contains__(name) == False:
            return None
        array = group[name][:]

        if group[name].attrs.get('dimensionality') is not None:
            return pq.Quantity(array, group[name].attrs['dimensionality'])

        dtype = None
        if array.dtype.type == np.void:
            array = self._decode_string_array(array)
            dtype = 'S'

        if group[name].attrs.get('is_list'):
            return array

        return np.array(array, dtype=dtype)

    def _decode_string_array(self, array):
        if len(np.shape(array)) == 0:
            return self._decode_string(array)

        result = []
        for row in array:
            result.append(self._decode_string_array(row))

        return result

    def _decode_string(self, string):
        return str(string.tostring().decode())
