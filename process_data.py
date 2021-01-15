from data_funcs import read_data, step_1, step_2, step_3, step_4

file = 'assignment1.csv'


def main():
    (read_data(file).pipe(step_1)
                    .pipe(step_2)
                    .pipe(step_3)
                    .pipe(step_4)
                    .to_csv('data.csv', index=False))


if __name__ == "__main__":
    main()
