/*-------------------------------------------------------------------------*/
/**
   @file common.c
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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>

#include "common.h"

/* matrix allocation */
void **makematr(int r, int c, int s) {
  int i;
  char **pc;
  short int **psi;
  int **pi;
  double **pd;

  switch(s) {
  case sizeof(char):
    pc=(char **)malloc(r*sizeof(char *));
    if(!pc) writelog(TRUE, MAKEMATR_RC, "error in makematr 1\n");
    for(i=0; i<r; i++) {
      pc[i]=(char *)malloc(c*sizeof(char));
      if(!pc[i]) writelog(TRUE, MAKEMATR_RC, "error in makematr 2\n");
      memset(pc[i],0,c*sizeof(char));
    }
    return (void **)pc;
  case sizeof(short int):
    psi=(short int **)malloc(r*sizeof(short int*));
    if(!psi) writelog(TRUE, MAKEMATR_RC, "error in makematr 3\n");
    for(i=0; i<r; i++) {
      psi[i]=(short int *)malloc(c*sizeof(short int));
      if(!psi[i]) writelog(TRUE, MAKEMATR_RC, "error in makematr 4\n");
      memset(psi[i],0,c*sizeof(short int));
    }
    return (void **)psi;
  case sizeof(int):
    pi=(int **)malloc(r*sizeof(int*));
    if(!pi) writelog(TRUE, MAKEMATR_RC, "error in makematr 5\n");
    for(i=0; i<r; i++) {
      pi[i]=(int *)malloc(c*sizeof(int));
      if(!pi[i]) writelog(TRUE, MAKEMATR_RC, "error in makematr 6\n");
      memset(pi[i],0,c*sizeof(int));
    }
    return (void **)pi;
  case sizeof(double):
    pd=(double **)malloc(r*sizeof(double*));
    if(!pd) writelog(TRUE, MAKEMATR_RC, "error in makematr 7 for %d rows\n",r);
    for(i=0; i<r; i++) {
      pd[i]=(double *)malloc(c*sizeof(double));
      if(!pd[i]) writelog(TRUE, MAKEMATR_RC, "error in makematr 8 for %d cols\n",c);
      memset(pd[i],0,c*sizeof(double));
    }
    return (void **)pd;
    break;
  default:
    writelog(TRUE,MAKEMATR_RC,"Unexpected size: %d\n",s);
    break;
  }
  return NULL;
}

/* vector allocation */
void *makevect(int d, int s) {
  char *pc;
  short int *psi;
  int *pi;
  double *pd;

  switch(s) {
  case sizeof(char):
    pc=(char *)malloc(d*sizeof(char));
    if(!pc) writelog(TRUE, MAKEVECT_RC, "error in makevect 1\n");
    memset(pc,0,d*sizeof(char));
    return (void *)pc;

  case sizeof(short int):
    psi=(short int *)malloc(d*sizeof(short int));
    if(!psi) writelog(TRUE, MAKEVECT_RC, "error in makevect 2\n");
    memset(psi,0,d*sizeof(short int));
    return (void *)psi;

  case sizeof(int):
    pi=(int *)malloc(d*sizeof(int));
    if(!pi) writelog(TRUE, MAKEVECT_RC, "error in makevect 3\n");
    memset(pi,0,d*sizeof(int));
    return (void *)pi;

  case sizeof(double):
    pd=(double *)malloc(d*sizeof(double));
    if(!pd) writelog(TRUE, MAKEVECT_RC, "error in makevect 4\n");
    memset(pd,0,d*sizeof(double));
    return (void *)pd;
    break;

  default:
    writelog(TRUE,MAKEVECT_RC,"Unexpected size: %d\n",s);
    break;
  }
  return NULL;
}

/* free matrix: r = number of rows */
void freematr(int r, void **matr) {
    int i;

    for (i=0; i<r; i++) {
        free(matr[i]);
    }
    free(matr);
}

/* free vect */
void freevect(void *vect) {
    free(vect);
}

void *Malloc(int size) {
  void *r;
  if(size<=0) {
    writelog(TRUE,MALLOC_RC,"malloc invalid size");
  }
  if((r=malloc(size))==NULL) {
    writelog(TRUE,MALLOC_RC,"malloc failed");
  }
  memset(r, 0, size);
  return r;
}

void Free(void *p) {
  free(p);
  p = NULL;
}

/* .............
   ..My strsep..
   ............. */

char *StrSep(char **inputstr, int ch) {
  char *app, *retstr;

  if ((*inputstr == NULL)||(**inputstr == '\0')) return(NULL);
  app = strchr(*inputstr, ch);
  retstr = *inputstr;
  if (app != NULL) {
    *inputstr = app + 1;
    *app = '\0';
  }
  else {
    *inputstr = NULL;
  }
  return(retstr);
}

char *StrStrSep(const char *inputstr, const char *source) {
  char *app, *final;
  int i;

  final = NULL;
  app = strstr(inputstr, source);
  if (app == NULL) {
    return(NULL);
  }
  for (i = 0; i < strlen(source); i++) {
    app++;
  }
  while ((app[0] == ' ') || (app[0] == '\t')) {
    app++;
  }
  if ((app[0] != '\n') && (app[0] != '\0')) {
    final = Strdup(app);
    app = final;
    while ((app[0] != '\0') && (app[0] != '\n')) {
      app++;
    }
    if (app[0] == '\n') {
      app[0] = '\0';
    }
  }
  else {
    return(NULL);
  }
  return(final);
}

char *Strdup(char *p) {
  char *r;
  if(p==NULL) {
    writelog(TRUE,STRDUP_RC,"strdup invalid pointer");
  }
  if((r=strdup(p))==NULL) {
    writelog(TRUE,STRDUP_RC,"strdup");
  }
  return r;
}

FILE *Fopen(const char *filename, const char *mode)
{
        FILE    *fp;

        if ( (fp = fopen(filename, mode)) == NULL)
          writelog(TRUE,FOPEN_RC," failed fopen for %s", filename);

        return(fp);
}

/* inner product (double) */
double inprodf(double *v, double *w, int dim) {
    int i;
    double sp = 0.0;

    for (i=0; i<dim; i++) {
        sp += v[i]*w[i];
    }
    return sp;
}


/* ............................................................................
   ............................................................................
   ............................................................................
   Compute escape sequences in string
............................................................................
............................................................................
............................................................................ */

void FixEscapeSeq(char *msg, int msgLen) {
        int i;


        for (i=0; i<msgLen; i++) {
                if (msg[i]== '\\') {
                        if (msgLen <= i+1)
                                break;
                        switch (msg[i+1]) {
                        case 'n':               /* New Line */
                                msg[i] = '\n';
                                memmove(msg+i+1, msg+i+2, msgLen - i - 2);
                                break;

                        case 't':               /* Tab */
                                msg[i] = '\t';
                                memmove(msg+i+1, msg+i+2, msgLen - i - 2);
                                break;

                        default:
                                writelog(TRUE,APPLICATION_RC,"System: LINUX\n", msg[i+1]  , msg);
                                break;
                        }
                }
        }

}

/* ............................................................................
   ............................................................................
   ............................................................................
   Get Strings for log messages: returns zero if no errors
............................................................................
............................................................................
............................................................................ */
#if defined(USEEXTMSG)
void GetLogMessages() {
        FILE *fp;
        char  line[MAXINPUTLINE];

        if(getenv(MSGFILENENV)) {
          fp = fopen(getenv(MSGFILENENV),"r");
        } else {
          fp = fopen(MSGFILE,"r");
        }
        if (!fp) {
                printf("Could not open messages file\n");
                exit(APPLICATION_RC);
        }


        while (fgets(line, MAXINPUTLINE, fp)!=NULL) {
                int num,numFields, numRead, msgLen;
                char *msg;


                numFields = sscanf(line, "%d\t", &num);
                /* fprintf(stderr,"found %d fields num=%d\n", numFields, num); */
                if (numFields<1) {
                        writelog(TRUE,APPLICATION_RC,"System: LINUX\n", line);
                        break;
                }

                /* Skip bytes for num and tab */
                numRead = snprintf(NULL, 0, "%d\t", num);
                msg = line + numRead;

                /* Skip first '"' and last '"'  */
                msg ++;

                msgLen = strlen(msg);   // This is newLine position
                msg[msgLen-2] = 0;              // This is '"' position


                /* ffprintf(stderr,"found msg (len=%d)%s\n", msgLen, msg); */
                FixEscapeSeq(msg, msgLen-1);
                /* ffprintf(stderr,"fixedMsg)%s\n", msg);       */

                logmessage[num] = Strdup(msg);
        }

        fclose(fp);
}
#endif

/* ............................................................................
   ............................................................................
   ............................................................................
   write log file
............................................................................
............................................................................
............................................................................ */
void writelog(int end, int rc, const char *fmt, ...) {
  static FILE *filelog=NULL;
  char buf[MAXSTRLEN+1];
  char LogFileName[]="NNML.log";
  va_list ap;
  va_start(ap, fmt);
#ifdef  HAVE_VSNPRINTF
  vsnprintf(buf, MAXSTRLEN, fmt, ap);       /* safe */
#else
  vsprintf(buf, fmt, ap);                   /* not safe */
#endif
  if(filelog==NULL) {
    filelog=fopen(LogFileName,"w");
    if(filelog==NULL) {
      fprintf(stderr,"Could not open %s file for logging, using stderr\n",LogFileName);
      filelog=stderr;
    }
  }
  fputs(buf,filelog);
  fflush(filelog);
  if(end) {
    fclose(filelog);
    exit(rc);
  }
}


