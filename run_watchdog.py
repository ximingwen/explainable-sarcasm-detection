import time
import os
import shutil
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class NewFileHandler(FileSystemEventHandler):
    def __init__(self, base_target_folder):
        super().__init__()
        self.target_folder = base_target_folder
        self.created_files = []
        self.set_counter = 60

        print(self.set_counter)

    def on_created(self, event):
        # This function is called when a new file is created
        if not event.is_directory:
            if (len(os.listdir(folder_to_monitor))!=0):
                time.sleep(10)
                file_path = event.src_path
                file_name = os.path.basename(file_path)
                print(f'New file detected: {file_name}')

                new_folder_path = os.path.join(self.target_folder, "epoch_"+str(self.set_counter))
                os.makedirs(new_folder_path, exist_ok=True)

                print(f'4 new files detected. Copying to {self.target_folder}...')

                for file_name in os.listdir(folder_to_monitor):
                    file = os.path.join(folder_to_monitor, file_name)

                    shutil.move(file, os.path.join(new_folder_path, file_name))
                    print(f'File {file_name} copied.')

                self.set_counter += 1

def start_monitoring(folder_to_monitor, folder_to_copy):
    path_to_watch = folder_to_monitor
    target_folder = folder_to_copy

    event_handler = NewFileHandler(target_folder)
    observer = Observer()
    observer.schedule(event_handler, path_to_watch, recursive=False)
    observer.start()

    print(f'Starting to monitor {path_to_watch} for new files...')

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        observer.join()

if __name__ == "__main__":
    # Set the folder to monitor and folder to copy new files to
    folder_to_monitor = './src/runs/10_31'
    folder_to_copy = './src/runs/PROTOCNN'

    start_monitoring(folder_to_monitor, folder_to_copy)