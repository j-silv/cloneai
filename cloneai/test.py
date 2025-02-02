import cloneai.data.split as split


def test_split():

    infile = "./data/test/in.aac"
    silences = split.get_silences(infile, silence_db=-30, min_silence_s=1.0, verbose=True)
    print(f"{silences=}")
    nonsilences = split.convert_silences_to_nonsilences(*silences, min_nonsilence_s=0.5)
    print(f"{nonsilences=}")

    for i, (start, end) in enumerate(nonsilences):
        outfile = f"./data/test/in_{i}.aac"
        split.write(infile, outfile, ss=start, to=end, verbose=True, sample_rate_hz=44100)

    # ----------------------------------------------------------------------------------------

    infile = "./data/test/scott.aac"
    silences = split.get_silences(infile, silence_db=-30, min_silence_s=1.0, verbose=True)
    # print(f"{silences=}")
    nonsilences = split.convert_silences_to_nonsilences(*silences, min_nonsilence_s=3.0)
    # print(f"{nonsilences=}")

    for i, (start, end) in enumerate(nonsilences):
        outfile = f"./data/test/scott_{i}.aac"
        split.write(infile, outfile, ss=start, to=end, verbose=True, sample_rate_hz=44100)

        if i == 10:
            break

    # ----------------------------------------------------------------------------------------




if __name__ == "__main__":
    test_split()
