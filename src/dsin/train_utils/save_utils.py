import datetime


class FileNameGenerator:
    @staticmethod
    def create_fname(fname_suffix: str) -> str:
        curr_datetime = str(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M'))
        return f'{curr_datetime}{fname_suffix}'




