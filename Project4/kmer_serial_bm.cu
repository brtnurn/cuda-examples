#include <iostream>
using namespace std;

#define ALPHABET_SIZE 4

int get_index(char c)
{
    switch (c)
    {
    case 'A':
        return 0;
    case 'C':
        return 1;
    case 'G':
        return 2;
    case 'T':
        return 3;
    default:
        return -1;
    }
}

void preprocess_bad_char(char *pattern, int k, int bad_char[])
{
    for (int i = 0; i < ALPHABET_SIZE; i++)
        bad_char[i] = -1;

    for (int i = 0; i < k; i++)
    {
        int idx = get_index(pattern[i]);
        if (idx != -1)
            bad_char[idx] = i;
    }
}

void kmer_serial_bm(char *ref, char *reads, int *counts, int ref_len, int read_len, int max_read_len, int read_count, int k)
{
    for (int i = 0; i < read_count; i++)
    {
        int count = 0;
        for (int l = 0; l < read_len - k + 1; l++)
        {
            int bad_char[ALPHABET_SIZE];
            preprocess_bad_char(&reads[i * max_read_len + l], k, bad_char);

            for (int j = 0; j <= ref_len - k;)
            {
                int r = k - 1;

                while (r >= 0 && ref[j + r] == reads[i * max_read_len + l + r])
                {
                    r--;
                }

                if (r < 0)
                {
                    count++;
                    j += 1;
                }
                else
                {
                    int idx = get_index(ref[j + r]);
                    int shift = (idx != -1) ? max(1, r - bad_char[idx]) : 1;
                    j += shift;
                }
            }
        }
        counts[i] = count;
    }
}
