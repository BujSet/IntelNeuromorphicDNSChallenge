import os
import glob
import torch
import numpy as np
import re
import soundfile as sf
from typing import Tuple, Dict, Any
import random

class DNSAudioAndCrepeCleanOnly:
    """Audio dataset loader for DNS to return clean speech samples along with CREPE pitch annotations.

    Parameters
    ----------
    root : str, optional
        Path of the dataset location, by default './'.
    maxFiles : int, optional
        Number of files to use as subset for faster training
    """
    def __init__(self, root: str = './', maxFiles: int = -1) -> None:
        self.root = root
        self.clean_files = glob.glob(root + 'clean/**.wav')
        self.crepe_files = glob.glob(root + 'crepe_pitch_annotations/clean/**.csv')
        # always coollect all crepe file , even if we are only looking at subset of clean files
        # because we do the fileid matching in the getitem method
        assert(len(self.crepe_files) >= len(self.clean_files))
        self.file_id_from_name = re.compile('fileid_(\d+)')
        if (maxFiles > len(self.clean_files)):
            print("Too many files to subsample dataset "+ str(maxFiles) + "/" + str(len(self.clean_files)))
            assert(False)

        # Don't do anything if param isnt set or if we're using the entire dataset
        if (maxFiles > 0 and maxFiles != len(self.clean_files)):
            randStart = random.randint(0, len(self.clean_files) - maxFiles - 1)
            assert(randStart + maxFiles <= len(self.clean_files))
            self.clean_files = self.clean_files[randStart:randStart+maxFiles]
            print("Using slice dataset[" + str(randStart) + ":" + str(randStart+maxFiles) + "] with "+str(len(self.clean_files)) + " samples")

    def _get_filenames(self, n: int) -> Tuple[str]:
        clean_file = self.clean_files[n % self.__len__()]
        return clean_file

    def __getitem__(self, n: int) -> Tuple[np.ndarray,
                                           np.ndarray,
                                           np.ndarray,
                                           np.ndarray,
                                           Dict[str, Any],
                                           int]:
        """Gets the nth sample from the dataset.

        Parameters
        ----------
        n : int
            Index of the dataset sample.

        Returns
        -------
        np.ndarray
            Clean audio sample.
        np.ndarray
            CREPE pitch prediction timestamps
        np.ndarray
            CREPE pitch prediction values
        np.ndarray
            CREPE pitch prediction confidences
        Dict
            Sample metadata.
        n
            Index of dataset sample
        """
        clean_file = self._get_filenames(n)
        filename = clean_file.split(os.sep)[-1]
        file_id = int(self.file_id_from_name.findall(filename)[0])
        clean_audio, sampling_frequency = sf.read(clean_file)
        num_samples = 30 * sampling_frequency  # 30 sec data
        metadata = {'fs': sampling_frequency}

        if len(clean_audio) > num_samples:
            clean_audio = clean_audio[:num_samples]
        else:
            clean_audio = np.concatenate([clean_audio,
                                          np.zeros(num_samples
                                                   - len(clean_audio))])
        # TODO read CREPE data
        crepe_times = None
        crepe_values = None
        crepe_confs = None
        crepe_file = "clean_fileid_" + str(file_id) + ".f0.csv"
        crepe_file = os.path.join("clean", crepe_file)
        crepe_file = os.path.join("crepe_pitch_annotations", crepe_file)
        crepe_file = os.path.join(self.root, crepe_file)
        with open(crepe_file, 'r') as crepe_f:
            lines = crepe_f.readlines()
            crepe_times = np.zeros(len(lines)-1, dtype=float)
            crepe_values =np.zeros(len(lines)-1, dtype=float)
            crepe_confs = np.zeros(len(lines)-1, dtype=float)
            # Skip header row
            for i in range(1, len(lines)):
                line = lines[i]
                values = line.split(",")
                assert(len(values) == 3)
                crepe_times[i-1] = float(values[0])
                crepe_values[i-1] = float(values[1])
                crepe_confs[i-1] = float(values[2])

        return clean_audio, crepe_times, crepe_values, crepe_confs, metadata, n

    def __len__(self) -> int:
        """Length of the dataset.
        """
        return len(self.clean_files)

    def collate_fn(self, batch):
        clean = []
        crepe_times = []
        crepe_vals = []
        crepe_confs = []

        indices = torch.IntTensor([s[5] for s in batch])

        for sample in batch:
            clean += [torch.FloatTensor(sample[0])]
            crepe_times += [torch.FloatTensor(sample[1])]
            crepe_vals += [torch.FloatTensor(sample[2])]
            crepe_confs += [torch.FloatTensor(sample[3])]

        return torch.stack(clean), torch.stack(crepe_times), torch.stack(crepe_vals), torch.stack(crepe_confs), indices

class DNSAudioCleanOnly:
    """Audio dataset loader for DNS to only return clean speech samples.

    Parameters
    ----------
    root : str, optional
        Path of the dataset location, by default './'.
    maxFiles : int, optional
        Number of files to use as subset for faster training
    """
    def __init__(self, root: str = './', maxFiles: int = -1) -> None:
        self.root = root
        self.clean_files = glob.glob(root + 'clean/**.wav')
        if (maxFiles > len(self.clean_files)):
            print("Too many files to subsample dataset "+ str(maxFiles) + "/" + str(len(self.clean_files)))
            assert(False)

        # Don't do anything if param isnt set or if we're using the entire dataset
        if (maxFiles > 0 and maxFiles != len(self.clean_files)):
            randStart = random.randint(0, len(self.clean_files) - maxFiles - 1)
            assert(randStart + maxFiles <= len(self.clean_files))
            self.clean_files = self.clean_files[randStart:randStart+maxFiles]
            print("Using slice dataset[" + str(randStart) + ":" + str(randStart+maxFiles) + "] with "+str(len(self.clean_files)) + " samples")

    def _get_filenames(self, n: int) -> Tuple[str]:
        clean_file = self.clean_files[n % self.__len__()]
        return clean_file

    def __getitem__(self, n: int) -> Tuple[np.ndarray,
                                           Dict[str, Any],
                                           int]:
        """Gets the nth sample from the dataset.

        Parameters
        ----------
        n : int
            Index of the dataset sample.

        Returns
        -------
        np.ndarray
            Clean audio sample.
        Dict
            Sample metadata.
        int
            index
        """
        clean_file= self._get_filenames(n)
        clean_audio, sampling_frequency = sf.read(clean_file)
        num_samples = 30 * sampling_frequency  # 30 sec data
        metadata = {'fs': sampling_frequency}

        if len(clean_audio) > num_samples:
            clean_audio = clean_audio[:num_samples]
        else:
            clean_audio = np.concatenate([clean_audio,
                                          np.zeros(num_samples
                                                   - len(clean_audio))])
        return clean_audio, metadata, n

    def __len__(self) -> int:
        """Length of the dataset.
        """
        return len(self.clean_files)

    def collate_fn(self, batch):
        clean = []

        indices = torch.IntTensor([s[2] for s in batch])

        for sample in batch:
            clean += [torch.FloatTensor(sample[0])]

        return torch.stack(clean), indices

class DNSAudioNoNoise:
    """Audio dataset loader for DNS to only return clean and noisy samples.

    Parameters
    ----------
    root : str, optional
        Path of the dataset location, by default './'.
    """
    def __init__(self, root: str = './', maxFiles: int = -1) -> None:
        self.root = root
        # Some of the noisy files has a non-standard character in the name:
        # 2
        # So when we do need to use the noisy dataset, it's better to just read 
        # the files names directly with glob.glob
        self.noisy_files = glob.glob(root + 'noisy/**.wav')
        if (maxFiles > len(self.noisy_files)):
            print("Too many files to subsample dataset "+ str(maxFiles) + "/" + str(len(self.noisy_files)))
            assert(False)

        # Don't do anything if param isnt set or if we're using the entire dataset
        if (maxFiles > 0 and maxFiles != len(self.noisy_files)):
            randStart = random.randint(0, len(self.noisy_files) - maxFiles - 1)
            assert(randStart + maxFiles <= len(self.noisy_files))
            self.noisy_files = self.noisy_files[randStart:randStart+maxFiles]
            print("Using slice dataset[" + str(randStart) + ":" + str(randStart+maxFiles) + "] with "+str(len(self.noisy_files)) + " samples")

        self.file_id_from_name = re.compile('fileid_(\d+)')
        self.snr_from_name = re.compile('snr(-?\d+)')
        self.target_level_from_name = re.compile('tl(-?\d+)')
        self.source_info_from_name = re.compile('^(.*?)_snr')

    def _get_filenames(self, n: int) -> Tuple[str, str, Dict[str, Any]]:
        noisy_file = self.noisy_files[n % self.__len__()]
        filename = noisy_file.split(os.sep)[-1]
        file_id = int(self.file_id_from_name.findall(filename)[0])
        clean_file = self.root + f'clean/clean_fileid_{file_id}.wav'
        snr = int(self.snr_from_name.findall(filename)[0])
        target_level = int(self.target_level_from_name.findall(filename)[0])
        source_info = self.source_info_from_name.findall(filename)[0]
        metadata = {'snr': snr,
                    'target_level': target_level,
                    'source_info': source_info}
        return clean_file, noisy_file, metadata

    def __getitem__(self, n: int) -> Tuple[np.ndarray,
                                           np.ndarray,
                                           Dict[str, Any],
                                           int]:
        """Gets the nth sample from the dataset.

        Parameters
        ----------
        n : int
            Index of the dataset sample.

        Returns
        -------
        np.ndarray
            Clean audio sample.
        np.ndarray
            Noisy audio sample.
        Dict
            Sample metadata.
        """
        clean_file, noisy_file, metadata = self._get_filenames(n)
        noisy_audio, sampling_frequency = sf.read(noisy_file)
        clean_audio, _ = sf.read(clean_file)
        num_samples = 30 * sampling_frequency  # 30 sec data
        metadata['fs'] = sampling_frequency

        if len(noisy_audio) > num_samples:
            noisy_audio = noisy_audio[:num_samples]
        else:
            noisy_audio = np.concatenate([noisy_audio,
                                          np.zeros(num_samples
                                                   - len(noisy_audio))])
        if len(clean_audio) > num_samples:
            clean_audio = clean_audio[:num_samples]
        else:
            clean_audio = np.concatenate([clean_audio,
                                          np.zeros(num_samples
                                                   - len(clean_audio))])
        return clean_audio, noisy_audio, metadata, n

    def __len__(self) -> int:
        """Length of the dataset.
        """
        return len(self.noisy_files)

    def collate_fn(self, batch):
        clean, noisy = [], []

        indices = torch.IntTensor([s[3] for s in batch])

        for sample in batch:
            clean += [torch.FloatTensor(sample[0])]
            noisy += [torch.FloatTensor(sample[1])]

        return torch.stack(clean), torch.stack(noisy), indices
    
class DNSAudioNoNoisy:
    """Audio dataset loader for DNS to only return clean and noise samples.

    Parameters
    ----------
    root : str, optional
        Path of the dataset location, by default './'.
    """
    def __init__(self, 
            root: str = './', 
            maxFiles: int = -1,
            noisyFileRoster: str = 'noisy_file_names.txt') -> None:
        self.root = root
        # Some of the noisy files has a non-standard character in the name:
        # 2
        # However, when we don't need to use the noisy dataset, we can rely 
        # on a raw text file that has all the  noisy file names. This works
        # because we don't need to actuall load the noisy files with the 
        # soundfile package, just extract information from the file name. So,
        # in this way, the noisy wav can be safely ignored.
        self.noisy_files = None
        roster = os.path.join(root, "noisy")
        roster = os.path.join(roster, noisyFileRoster)
        with open(roster, "r") as fp:
            self.noisy_files = [line.rstrip('\n') for line in fp.readlines()]
        self.noisy_files = [filename for filename in self.noisy_files if not filename.endswith('.txt')]
        assert(self.noisy_files != None)
        assert(not "noisy_file_names.txt" in self.noisy_files)
        assert(not noisyFileRoster in self.noisy_files)
        if (maxFiles > len(self.noisy_files)):
            print("Too many files to subsample dataset "+ str(maxFiles) + "/" + str(len(self.noisy_files)))
            assert(False)

        # Don't do anything if param isnt set or if we're using the entire dataset
        if (maxFiles > 0 and maxFiles < len(self.noisy_files)):
            self.noisy_files = random.sample(self.noisy_files, maxFiles)
        else:
            print(f"Warning: maxFiles set to invlaid value of {maxFiles}, no sampling performed")
        
        self.file_id_from_name = re.compile('fileid_(\d+)')
        self.snr_from_name = re.compile('snr(-?\d+)')
        self.target_level_from_name = re.compile('tl(-?\d+)')
        self.source_info_from_name = re.compile('^(.*?)_snr')

    def _get_filenames(self, n: int) -> Tuple[str, str, Dict[str, Any]]:
        noisy_file = self.noisy_files[n % self.__len__()]
        filename = noisy_file.split(os.sep)[-1]
        file_id = int(self.file_id_from_name.findall(filename)[0])
        clean_file = self.root + f'clean/clean_fileid_{file_id}.wav'
        noise_file = self.root + f'noise/noise_fileid_{file_id}.wav'
        snr = int(self.snr_from_name.findall(filename)[0])
        target_level = int(self.target_level_from_name.findall(filename)[0])
        source_info = self.source_info_from_name.findall(filename)[0]
        metadata = {'snr': snr,
                    'target_level': target_level,
                    'source_info': source_info}
        return clean_file, noise_file, metadata

    def __getitem__(self, n: int) -> Tuple[np.ndarray,
                                           np.ndarray,
                                           Dict[str, Any],
                                           int]:
        """Gets the nth sample from the dataset.

        Parameters
        ----------
        n : int
            Index of the dataset sample.

        Returns
        -------
        np.ndarray
            Clean audio sample.
        np.ndarray
            Noise audio sample.
        Dict
            Sample metadata.
        """
        clean_file, noise_file, metadata = self._get_filenames(n)
        clean_audio, sampling_frequency = sf.read(clean_file)
        noise_audio, _ = sf.read(noise_file)
        num_samples = 30 * sampling_frequency  # 30 sec data
        metadata['fs'] = sampling_frequency

        if len(clean_audio) > num_samples:
            clean_audio = clean_audio[:num_samples]
        else:
            clean_audio = np.concatenate([clean_audio,
                                          np.zeros(num_samples
                                                   - len(clean_audio))])
        if len(noise_audio) > num_samples:
            noise_audio = noise_audio[:num_samples]
        else:
            noise_audio = np.concatenate([noise_audio,
                                          np.zeros(num_samples
                                                   - len(noise_audio))])
        return clean_audio, noise_audio, metadata, n

    def __len__(self) -> int:
        """Length of the dataset.
        """
        return len(self.noisy_files)

    def collate_fn(self, batch):
        clean, noise = [], []

        indices = torch.IntTensor([s[3] for s in batch])

        for sample in batch:
            clean += [torch.FloatTensor(sample[0])]
            noise += [torch.FloatTensor(sample[1])]

        return torch.stack(clean), torch.stack(noise), indices

class DNSAudioCleanAndPitch:
    """
        Audio dataset loader for DNS to return clean speech samples. Also 
        returns fundamental frequency estimation read from files

    Parameters
    ----------
    root : str, optional
        Path of the dataset location, by default './'.
    maxFiles : int, optional
        Number of files to use as subset for faster training
    n_fft : int, optional
        Used for shaping the output to match fft of external values
    """
    def __init__(self, root: str = './', maxFiles: int = -1, n_fft: int = 512) -> None:
        self.root = root
        self.clean_files = glob.glob(root + 'clean/**.wav')
        if (maxFiles > len(self.clean_files)):
            print("Too many files to subsample dataset "+ str(maxFiles) + "/" + str(len(self.clean_files)))
            assert(False)

        # Don't do anything if param isnt set or if we're using the entire dataset
        if (maxFiles > 0 and maxFiles != len(self.clean_files)):
            randStart = random.randint(0, len(self.clean_files) - maxFiles - 1)
            assert(randStart + maxFiles <= len(self.clean_files))
            self.clean_files = self.clean_files[randStart:randStart+maxFiles]
            print("Using slice dataset[" + str(randStart) + ":" + str(randStart+maxFiles) + "] with "+str(len(self.clean_files)) + " samples")
        self.num_fft_frames = int((30 * 16000 / (n_fft // 4)) + 1)

    def _get_filenames(self, n: int) -> Tuple[str]:
        clean_file = self.clean_files[n % self.__len__()]
        return clean_file

    def __getitem__(self, n: int) -> Tuple[np.ndarray,
                                           Dict[str, Any],
                                           Dict[str, Any],
                                           int]:
        """Gets the nth sample from the dataset.

        Parameters
        ----------
        n : int
            Index of the dataset sample.

        Returns
        -------
        np.ndarray
            Clean audio sample.
        np.ndarray
            CREPE times array for fundamental frequency estiamtions.
        np.ndarray
            CREPE frequency array for fundamental frequency estiamtions.
        np.ndarray
            CREPE confidence array for fundamental frequency estiamtions.
        Dict
            Sample metadata.
        int
            index
        """
        clean_file= self._get_filenames(n)
        clean_audio, sampling_frequency = sf.read(clean_file)
        num_samples = 30 * sampling_frequency  # 30 sec data
        metadata = {'fs': sampling_frequency}

        if len(clean_audio) > num_samples:
            clean_audio = clean_audio[:num_samples]
        else:
            clean_audio = np.concatenate([clean_audio,
                                          np.zeros(num_samples
                                                   - len(clean_audio))])
        pathTokens = clean_file.split("/")
        crepeFilePath = os.path.join(self.root, "crepe_pitch_annotations")
        crepeFilePath = os.path.join(crepeFilePath, "clean")
        crepeFileName = pathTokens[-1].replace(".wav", ".f0.csv")
        crepeFile = os.path.join(crepeFilePath, crepeFileName)
        with open(crepeFile) as f:
            lines = f.readlines()
            # Skip header line
            times = []
            freqs = []
            confs = []
            for i in range(1, len(lines)):
                values = lines[i].split(",")
                assert(len(values) == 3)
                times.append(values[0])
                freqs.append(values[1])
                confs.append(values[2])
        times = np.array(times[0:self.num_fft_frames], dtype=np.float32)
        freqs = np.array(freqs[0:self.num_fft_frames], dtype=np.float32)
        confs = np.array(confs[0:self.num_fft_frames], dtype=np.float32)
        return clean_audio, times, freqs, confs, metadata, n

    def __len__(self) -> int:
        """Length of the dataset.
        """
        return len(self.clean_files)

    def collate_fn(self, batch):
        clean = []
        crepeTimes = []
        crepeFreqs = []
        crepeConfs = []

        indices = torch.IntTensor([s[5] for s in batch])

        for sample in batch:
            clean += [torch.FloatTensor(sample[0])]
            crepeTimes += [torch.FloatTensor(sample[1])]
            crepeFreqs += [torch.FloatTensor(sample[2])]
            crepeConfs += [torch.FloatTensor(sample[3])]

        clean = torch.stack(clean)
        crepeTimes = torch.stack(crepeTimes)
        crepeFreqs = torch.stack(crepeFreqs)
        crepeConfs = torch.stack(crepeConfs)
        return clean, crepeTimes, crepeFreqs, crepeConfs, indices

class DNSAudio:
    """Audio dataset loader for DNS.

    Parameters
    ----------
    root : str, optional
        Path of the dataset location, by default './'.
    maxFiles : int, optional
        Number of files to use as subset for faster training
    """
    def __init__(self, root: str = './', maxFiles: int = -1) -> None:
        self.root = root
        self.noisy_files = glob.glob(root + 'noisy/**.wav')
        if (maxFiles > len(self.noisy_files)):
            print("Too many files to subsample dataset "+ str(maxFiles) + "/" + str(len(self.noisy_files)))
            assert(False)

        # Don't do anything if param isnt set or if we're using the entire dataset
        if (maxFiles > 0 and maxFiles != len(self.noisy_files)):
            randStart = random.randint(0, len(self.noisy_files) - maxFiles - 1)
            assert(randStart + maxFiles <= len(self.noisy_files))
            self.noisy_files = self.noisy_files[randStart:randStart+maxFiles]
            print("Using slice dataset[" + str(randStart) + ":" + str(randStart+maxFiles) + "] with "+str(len(self.noisy_files)) + " samples")

        self.file_id_from_name = re.compile('fileid_(\d+)')
        self.snr_from_name = re.compile('snr(-?\d+)')
        self.target_level_from_name = re.compile('tl(-?\d+)')
        self.source_info_from_name = re.compile('^(.*?)_snr')

    def _get_filenames(self, n: int) -> Tuple[str, str, str, Dict[str, Any]]:
        noisy_file = self.noisy_files[n % self.__len__()]
        filename = noisy_file.split(os.sep)[-1]
        file_id = int(self.file_id_from_name.findall(filename)[0])
        clean_file = self.root + f'clean/clean_fileid_{file_id}.wav'
        noise_file = self.root + f'noise/noise_fileid_{file_id}.wav'
        snr = int(self.snr_from_name.findall(filename)[0])
        target_level = int(self.target_level_from_name.findall(filename)[0])
        source_info = self.source_info_from_name.findall(filename)[0]
        metadata = {'snr': snr,
                    'target_level': target_level,
                    'source_info': source_info}
        return noisy_file, clean_file, noise_file, metadata

    def __getitem__(self, n: int) -> Tuple[np.ndarray,
                                           np.ndarray,
                                           np.ndarray,
                                           Dict[str, Any],
                                           int]:
        """Gets the nth sample from the dataset.

        Parameters
        ----------
        n : int
            Index of the dataset sample.

        Returns
        -------
        np.ndarray
            Noisy audio sample.
        np.ndarray
            Clean audio sample.
        np.ndarray
            Noise audio sample.
        Dict
            Sample metadata.
        """
        noisy_file, clean_file, noise_file, metadata = self._get_filenames(n)
        noisy_audio, sampling_frequency = sf.read(noisy_file)
        clean_audio, _ = sf.read(clean_file)
        noise_audio, _ = sf.read(noise_file)
        num_samples = 30 * sampling_frequency  # 30 sec data
        metadata['fs'] = sampling_frequency

        if len(noisy_audio) > num_samples:
            noisy_audio = noisy_audio[:num_samples]
        else:
            noisy_audio = np.concatenate([noisy_audio,
                                          np.zeros(num_samples
                                                   - len(noisy_audio))])
        if len(clean_audio) > num_samples:
            clean_audio = clean_audio[:num_samples]
        else:
            clean_audio = np.concatenate([clean_audio,
                                          np.zeros(num_samples
                                                   - len(clean_audio))])
        if len(noise_audio) > num_samples:
            noise_audio = noise_audio[:num_samples]
        else:
            noise_audio = np.concatenate([noise_audio,
                                          np.zeros(num_samples
                                                   - len(noise_audio))])
        return noisy_audio, clean_audio, noise_audio, metadata, n

    def __len__(self) -> int:
        """Length of the dataset.
        """
        return len(self.noisy_files)

    def collate_fn(self, batch):
        noisy, clean, noise = [], [], []

        indices = torch.IntTensor([s[4] for s in batch])

        for sample in batch:
            noisy += [torch.FloatTensor(sample[0])]
            clean += [torch.FloatTensor(sample[1])]
            noise += [torch.FloatTensor(sample[2])]

        return torch.stack(noisy), torch.stack(clean), torch.stack(noise), indices

if __name__ == '__main__':
    train_set = DNSAudio(
        root='../../data/MicrosoftDNS_4_ICASSP/training_set/')
    validation_set = DNSAudio(
        root='../../data/MicrosoftDNS_4_ICASSP/validation_set/')
