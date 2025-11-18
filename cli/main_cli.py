import click
import file_input_cli as fic
import optical_flow_cli as opt
import visualization_cli as v

def get_video():
    """
    This function gets the path to the file

    Arguments: None

    Returns: user path
    """
    path = click.prompt('Enter your file path name', type=click.Path(
                                                                        exists=True,
                                                                        file_okay=True,
                                                                        dir_okay=False,
                                                                        resolve_path=True))
    
    return path



def main():
    (path) = get_video()
    my_video = fic.init_tiff_class(path)
    #fic.preprocess_tiff(my_video, preprocess)



if __name__ == "__main__":
    main()

