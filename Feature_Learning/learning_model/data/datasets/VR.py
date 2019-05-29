# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import glob
import re
import xml.dom.minidom as XD
import os.path as osp

from .bases import BaseImageDataset


class VR(BaseImageDataset):
    """
    VR

    Dataset statistics:

    """
    dataset_dir = 'VR'
    dataset_dir_test = './data/VeRi'

    def __init__(self, root='./data', verbose=True, **kwargs):
        super(VR, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'image_train/')
        self.query_dir = osp.join(self.dataset_dir_test, 'image_query/')
        self.gallery_dir = osp.join(self.dataset_dir_test, 'image_test/')
        #self.query_dir = osp.join(self.dataset_dir, 'image_query/')
        #self.gallery_dir = osp.join(self.dataset_dir, 'image_test/')
        self._check_before_run()

        train = self._process_dir(self.train_dir, relabel=True)
        query = self._process_dir_test(self.query_dir, relabel=False)
        gallery = self._process_dir_test(self.gallery_dir, relabel=False)
        #query = self._process_dir_demo(self.query_dir, relabel=False)
        #gallery = self._process_dir_demo(self.gallery_dir, relabel=False)
        if verbose:
            print("=> VR loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, dir_path, relabel=False):
        xml_dir =osp.join('./data/VR', 'train_label.xml')
        info = XD.parse(xml_dir).documentElement.getElementsByTagName('Item')
        
        pid_container = set()       
        for element in range(len(info)):
            pid = int(info[element].getAttribute('vehicleID'))
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}       
        
        dataset = []
        for element in range(len(info)):
            pid, camid = map(int, [info[element].getAttribute('vehicleID'), info[element].getAttribute('cameraID')[1:]])
            image_name = str(info[element].getAttribute('imageName'))
            if pid == -1: continue  # junk images are just ignored
            assert 0 <= pid <= 1501  # pid == 0 means background
            assert 1 <= camid <= 40
            camid -= 1  # index starts from 0
            if relabel: pid = pid2label[pid]
            dataset.append((osp.join(dir_path, image_name), pid, camid))

        return dataset
        
    def _process_dir_test(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d\d\d)')
        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            assert 0 <= pid <= 1000  # pid == 0 means background
            assert 1 <= camid <= 36
            camid -= 1  # index starts from 0
            if relabel: pid = pid2label[pid]
            dataset.append((img_path, pid, camid))

        return dataset
        
    def _process_dir_demo(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        img_paths.sort()
        pid_container = set()
        for img_path in img_paths:
            #pid, _ = map(int, pattern.search(img_path).groups())
            pid = 1
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in img_paths:
            pid, camid = 1, 2
            if pid == -1: continue  # junk images are just ignored
            assert 0 <= pid <= 1000  # pid == 0 means background
            assert 1 <= camid <= 36
            camid -= 1  # index starts from 0
            if relabel: pid = pid2label[pid]
            dataset.append((img_path, pid, camid))

        return dataset
