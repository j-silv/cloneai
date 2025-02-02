import cloneai.data.split as split


def test_split():

    infile = "./data/test/in.aac"
    silences = split.get_silences(infile, silence_db=-30, min_silence_s=1.0, verbose=True)
    print(f"{silences=}")
    nonsilences = split.convert_silences_to_nonsilences(*silences, min_nonsilence_s=0.5)
    print(f"{nonsilences=}")

    for i, (start, end) in enumerate(nonsilences):
        outfile = f"./data/test/out_{i}.aac"
        split.write(infile, outfile, ss=start, to=end, verbose=True, sample_rate_hz=44100)

    # ----------------------------------------------------------------------------------------

    # infile = "./data/test/scott.aac"
    # split.get_silences(infile, silence_db=-30, min_silence_s=1.0, verbose=True))



if __name__ == "__main__":
    test_split()
