import os

#TO DO, we may want to use a different form of config file. 

class ExtractorConfig(object):
    def __init__(self) -> None:
        super().__init__()
        self.config = {}
        self.config['dmap'] = None
        self.config['--pdb'] = None
        self.config['--chains'] = None
        self.config['--output'] = None
        self.config['--measure'] = 'CA'
        self.config['--mask-thresh'] = None

        self.config['--plaintext'] = None
        self.config['--asymmetric'] = None
        self.config['--title'] = None
        self.config['--xlabel'] = None
        self.config['--ylabel'] = None
        self.config['--font-family'] = 'sans'
        self.config['--font-size'] = 10
        self.config['--width-inches'] = 6.0
        self.config['--height-inches'] = 6.0
        self.config['--dpi'] = 80
        self.config['--greyscale'] = None
        self.config['--no-colorbar'] = None
        self.config['--transparent'] = None
        self.config['--show-frame'] = None
        self.config['--verbose'] = None


