class Tiff:
    """
    This is a class that imports TIFF file to program, converts TIFF to numpy array using TIFFFILE,
    and stores the video type
    """
    def __init__ (self, path, vid_type):
        """
        Initializes the class, saving the video path to the class as well as the video type
        """
        self.path = path
        self.vidtype = vid_type
    
    def import_vid (self, path):
        """
        Imports the TIFF file to the program

        Args:
        path = string that is a video path

        Returns:
        The opened file

        Preconditions:
        The path is a valid path to a TIFF video file
        """
        raise NotImplementedError
    
    def convert_to_numpy (self):
        """
        Converts the TIFF file to a numpy array using TIFFFILE

        Args:
        None

        Returns:
        numpy array

        Preconditions:
        Will have to call import_vid in this function
        """
        raise NotImplementedError
    