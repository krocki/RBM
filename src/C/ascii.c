#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>

main(int argc, char *argv[])
{
    int i=33, hex=0, extended=0;

    while (optind < argc) {
        switch (getopt(argc, argv, "dehxf")) {
        case 'd': /* decimal */
            hex=0;
            break;
        case 'e': /* extended vt100 chars */
            extended=1;
            break;
        case 'h': /* help screen */
            printf("%s usage:\n",argv[0]);
            printf("\t%s [ -d | -x ] [ -f ] [ -e ]\n", argv[0]);
            printf("\t-d - print out index in decimal\n");
            printf("\t-x - print out index in hexidecimal\n");
            printf("\t-f - full printout - don't skip control characters\n");
            printf("\t-e - Use ^N and ^O to enable some vt100 graphics\n");
            exit(0);
            break;
        case 'x': /* hex */
            hex=1;
            break;
        case 'f': /* full */
            i=0;
            break;
        }
    }
    
    /* i is set above */
    for (;i <= 255; i++) {
        /* print the number of the char, and the char */
        if (hex)
            printf(" %x ", i);
        else
            printf(" %d ", i);
        if (extended)
            printf("\xe%c\xf\t", i);
        else
            printf("%c\t", i);
        /* wrap to next line after 8 chars */
        if (i % 8 == 0) puts("");
    }
    puts("");
}
