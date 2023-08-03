if __name__ == '__main__':
    from ex_darpa import TC2DB

    # Build a standard dashbaord object
    dash_object = TC2DB().assemble_basedash()

    # uncomment to produces images for all frames
    # dash_object.visualize()

    # uncomment to produces images for n_th frame
    # n = 10
    # dash_object.visualize(frame_ind = n, show = True)

    # uncomment to produces image for last frame
    # dash_object.visualize_most_recent(show = True)

    # Visualize during optimization
    # dash_object.visualize_auto_refresh()

    # uncomment to make movie
    dash_object.visualize_all()
    # dash_object.make_mov()

    # uncomment to run gui
    # dash_object.run_GUI()