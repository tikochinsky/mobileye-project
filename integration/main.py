from controller import Controller


def main():
    pls_path = 'frames.pls'
    tfl_controller = Controller(pls_path)
    tfl_controller.run()


if __name__ == '__main__':
    main()
