#include <iostream>

using namespace std;

void kmer_serial(char *ref, char *reads, int *counts, int ref_len, int read_len, int max_read_len, int read_count, int k)
{
    for (int i = 0; i < read_count; i++)
    {
        int count = 0;
        for (int l = 0; l < read_len - k + 1; l++)
        {
            for (int j = 0; j < ref_len - k + 1; j++)
            {
                bool flag = true;
                for (int r = 0; r < k; r++)
                {
                    if (ref[j + r] != reads[i * max_read_len + l + r])
                    {
                        flag = false;
                        break;
                    }
                }
                if (flag)
                {
                    count++;
                }
            }
        }
        counts[i] = count;
    }
}
