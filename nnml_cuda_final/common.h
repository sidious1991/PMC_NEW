/*-------------------------------------------------------------------------*/
/**
   @file common.h
   @author IAC-CNR
   @date Dic 2009
   @brief Libreria di funzioni generiche

   Questo modulo implementa una collezione di funzioni generiche
   utilizzate negli altri moduli che compongono il programma.
*/
/*--------------------------------------------------------------------------*/
/************************************
 * Module: common.c
 *
 * COMMON PROCEDURES AND VARIABLES
 *
 ***********************************/
#if !defined(COMMON_H)
#define COMMON_H

#include <stdio.h>

#ifdef _MSC_VER
#define snprintf _snprintf
#endif

typedef int BOOL_T;
typedef double REAL;
#define REAL_T REAL

#if !defined(TRUE)
#define TRUE 1
#define FALSE 0
#endif

#define MMAX(i,j) ((i)>(j) ? i : j)
#define MMIN(i,j) ((i)>(j) ? j : i)
#define KRO(i,j) ((i)==(j) ? 1:0)
#define ODD(a) (((a)+2) % 2)
#define EVEN(a) (!ODD((a)))

void **makematr(int r, int c, int s);
void *makevect(int d, int s);
void freematr(int r, void **matr);
void freevect(void *vect);
char *StrSep(char **inputstr, int ch);
char *StrStrSep(const char *inputstr, const char *source);
char *Strdup(char *s);
void *Malloc(int s);
void Progress(int t, int Total);
FILE *Fopen(const char *filename, const char *mode);
double inprodf(double *v, double *w, int dim);

/* ............................................................................
   ............................................................................
   ............................................................................
   write log file
............................................................................
............................................................................
............................................................................ */
extern char *logmessage[];
void writelog(int, int, const char *, ...);
void GetLogMessages();

#if!defined(TRUE)
enum {FALSE, TRUE};
#endif

#define SKIPBLANK         while(po[0] & (po[0]==' ' || po[0]=='\t')) po++; \
                          if (!po[0]) { \
                             i++;      \
                             po=argv[i]; \
                          }
enum RC{OK, APPLICATION_RC, STRDUP_RC, MALLOC_RC, FOPEN_RC, POPEN_RC, FGETS_RC,
        SOCKET_RC, SEND_RC, RECV_RC, LOCK_RC, UNLOCK_RC, MAKEMATR_RC,
        MAKEVECT_RC, FINDLOG_RC, FREAD_RC, FSTAT_RC, MMAP_RC,
        CHECKSFAILURE, TOOMANYATF};

#define MAXINPUTLINE 10000
#define MAXSTRLEN 1024
#define MAXFILELENGTH 256
#define INVALID_INT -999999
#define INVALID_REAL -1.e7
#define INVALID_STR NULL
#define DEFMONTHS 60
#define DEFYEARS  30



#define READINTFI(v,s) snprintf(key,sizeof(key),"%s:%s","INPUTDATA",(s)); \
                          (v)=iniparser_getint(ini, key, INVALID_INT); \
                          if((v)==INVALID_INT) { \
                            writelog(TRUE,APPLICATION_RC, \
                                     "Invalid value for key <%s> from input file %s\n", \
                                     key, inputfile); \
                          }

#define READREALFI(v,s)  snprintf(key,sizeof(key),"%s:%s","INPUTDATA",(s)); \
                            (v)=(REAL)iniparser_getdouble(ini, key, INVALID_REAL);\
                            if((v)==(REAL)INVALID_REAL) { \
                              writelog(TRUE,APPLICATION_RC,\
                                       "Invalid value for key <%s> from input file %s\n",\
                                       key, inputfile); \
                            }

#define READSTRFI(v,s)  snprintf(key,sizeof(key),"%s:%s","INPUTDATA",(s)); \
                           char* temp=iniparser_getstring(ini, key, INVALID_STR);\
                           if(temp==INVALID_STR) { \
                                  writelog(TRUE,APPLICATION_RC,\
                                        "Invalid value for key <%s> from input file %s\n",\
                                        key, inputfile); \
                           }else{ strcpy((v),temp); /* sscanf(temp,"%s",(v)); */ }



#endif
