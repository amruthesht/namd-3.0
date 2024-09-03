/**
***  Copyright (c) 1995, 1996, 1997, 1998, 1999, 2000 by
***  The Board of Trustees of the University of Illinois.
***  All rights reserved.
**/

/*
   global functions as declared in common.h
*/

#if !defined(WIN32) || defined(__CYGWIN__)
#include <unistd.h>
#endif
#include <errno.h>
#include <string.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <ctype.h>

#include "common.h"
#include "Node.h"
#include "SimParameters.h"
#include "InfoStream.h"

#include "charm++.h"

#ifdef WIN32
#include <io.h>
#define access(PATH,MODE) _access(PATH,00)
#define NOCOMPRESSED
#endif

#ifdef USESTDIOSTUBS

// On the ASCI-Red, these functions have prototypes that don't match
// the standard, so I'll override them

int write(int fd, const void* buf, int count) 
{ 
  return write(fd,(void*)buf,(size_t)count);
}

int stat(const char* fn, struct stat* buf)
{
  return stat((char*)fn, buf);
}

#endif


const char *NAMD_version_string() {
  return NAMD_VERSION;  // macro defined in toplevel Makefile
}

// Query for Tcl-style version numbering in NAMD
void NAMD_version(int *major, int *minor, int *patchLevel, int *releaseType)
{
  int NAMD_major = 0;
  int NAMD_minor = 0;
  int NAMD_patchLevel = 0;
  int NAMD_releaseType = NAMD_UNKNOWN_RELEASE;

  // have to parse the version string to determine numbers
  const char *vstr = NAMD_version_string();
  char dot[4] = "";  // expect to read "."
  char release[8] = "";  // expect to read "" or "."
                         // or "alpha" or "beta" or "final" or "release"
  char extra[4] = "";  // bad version string if we read an extra character
  int n = sscanf(vstr, "%d%3[^0-9]%d%7[^0-9]%d%c",
      &NAMD_major, dot, &NAMD_minor, release, &NAMD_patchLevel, extra);
  if (n < 3 || n > 5) {
    NAMD_bug("Unable to parse NAMD_VERSION from Makefile");
  }
  if (n >= 4) {
    if (strcmp(release, "a") == 0 || strcmp(release, "alpha") == 0) {
      NAMD_releaseType = NAMD_ALPHA_RELEASE;
    }
    else if (strcmp(release, "b") == 0 || strcmp(release, "beta") == 0) {
      NAMD_releaseType = NAMD_BETA_RELEASE;
    }
    else if (strcmp(release, ".") == 0 ||
             strcmp(release, "final") == 0 ||
             strcmp(release, "release") == 0 ||
             strcmp(release, "patch") == 0) {
      NAMD_releaseType = NAMD_FINAL_RELEASE;
    }
    else {
      NAMD_bug("Unable to parse NAMD_VERSION from Makefile");
    }
  }
  else {
    NAMD_releaseType = NAMD_FINAL_RELEASE;
  }
  // now return values to caller
  if (major != NULL) *major = NAMD_major;
  if (minor != NULL) *minor = NAMD_minor;
  if (patchLevel != NULL) *patchLevel = NAMD_patchLevel;
  if (releaseType != NULL) *releaseType = NAMD_releaseType;
}


int NAMD_gcd(int a, int b)
{
   if(a == 0 || b == 0) return 0;
   else if(a == b) return a;
   else if(a > b) return NAMD_gcd(a-b,b);
   return NAMD_gcd(a, b-a);
}

// make a duplicate of a string
char *NAMD_stringdup(const char *s) {
  char *rs;

  if(!s)
    return NULL;

  rs = new char[strlen(s) + 1];
  strcpy(rs,s);

  return rs;
}


// signal all nodes, it's time to quit
void NAMD_quit(const char *err_msg)
{
  if ( ! err_msg ) err_msg = "(unknown error)";
  CkPrintf("EXITING: %s\n", err_msg);
  fflush(stdout);
  char repstr[24] = "";
  if (CmiNumPartitions() > 1
      || Node::Object()->simParameters->CUDASOAintegrateMode) {
    sprintf(repstr,"REPLICA %d ", CmiMyPartition());
    // CkAbort ensures that all replicas die
    CkAbort("%sEXITING: %s\n", repstr, err_msg);
  }
  CkError("%sEXITING: %s\n", repstr, err_msg);
#if CHARM_VERSION < 61000
  CkExit();
#else
  CkExit(0);
#endif
}

 
// signal all nodes, it's time to quit
void NAMD_die(const char *err_msg)
{
  if ( ! err_msg ) err_msg = "(unknown error)";
  CkPrintf("FATAL ERROR: %s\n", err_msg);
  fflush(stdout);
  char repstr[24] = "";
  if (CmiNumPartitions() > 1 || 
      (Node::Object()->simParameters != 0 &&
       Node::Object()->simParameters->CUDASOAintegrateMode)) {
    sprintf(repstr,"REPLICA %d ", CmiMyPartition());
    // CkAbort ensures that all replicas die
    CkAbort("%sFATAL ERROR: %s\n", repstr, err_msg);
  }
  CkError("%sFATAL ERROR: %s\n", repstr, err_msg);
#if CHARM_VERSION < 61000
  CkExit();
#else
  CkExit(1);
#endif
}


// signal all nodes, it's time to quit
void NAMD_err(const char *err_msg)
{
  if ( ! err_msg ) err_msg = "(unknown error)";
  const char *sys_err_msg = strerror(errno);
  if ( ! sys_err_msg ) sys_err_msg = "(unknown error)";
  CkPrintf("FATAL ERROR: %s: %s\n", err_msg, sys_err_msg);
  fflush(stdout);
  char repstr[24] = "";
  if (CmiNumPartitions() > 1 ||
      (Node::Object()->simParameters != 0 &&
       Node::Object()->simParameters->CUDASOAintegrateMode)) {
    sprintf(repstr,"REPLICA %d ", CmiMyPartition());
    // CkAbort ensures that all replicas die
    CkAbort("%sFATAL ERROR: %s: %s\n", repstr, err_msg, sys_err_msg);
  }
  CkError("%sFATAL ERROR: %s: %s\n", repstr, err_msg, sys_err_msg);
#if CHARM_VERSION < 61000
  CkExit();
#else
  CkExit(1);
#endif
}


// signal all nodes, it's time to quit and it's our own damn fault
void NAMD_bug(const char *err_msg)
{
  if ( ! err_msg ) err_msg = "(unknown error)";
  const char *bug_msg = 
    "FATAL ERROR: See http://www.ks.uiuc.edu/Research/namd/bugreport.html";
  CkPrintf("FATAL ERROR: %s\n%s\n",err_msg, bug_msg);
  fflush(stdout);
  char repstr[24] = "";
  if (CmiNumPartitions() > 1 ||
      (Node::Object()->simParameters != 0 &&
       Node::Object()->simParameters->CUDASOAintegrateMode)) {
    sprintf(repstr,"REPLICA %d ", CmiMyPartition());
    // CkAbort ensures that all replicas die
    CkAbort("%sFATAL ERROR: %s\n%s\n", repstr, err_msg, bug_msg);
  }
  CkError("%sFATAL ERROR: %s\n%s\n", repstr, err_msg, bug_msg);
#if CHARM_VERSION < 61000
  CkExit();
#else
  CkExit(2);
#endif
}


int NAMD_file_exists(const char *filename) {
  int rval;
  do {
    rval = access(filename, F_OK);
  } while ( rval != 0 && errno == EINTR );
  if ( rval != 0 && errno != ENOENT ) {
    const char *sys_err_msg = strerror(errno);
    if ( ! sys_err_msg ) sys_err_msg = "(unknown error)";
    iout << iERROR << "Error on checking file "
      << filename << ": " << sys_err_msg << "\n" << endi;
    fflush(stdout);
  }
  return ! rval;
}

// move filename to filename.BAK
void NAMD_backup_file(const char *filename, const char *extension)
{
  if (NAMD_file_exists(filename)) {
    if ( ! extension ) extension = ".BAK";
    char *backup = new char[strlen(filename)+strlen(extension)+1];
    strcpy(backup, filename);
    strcat(backup, extension);
#if defined(WIN32) && !defined(__CYGWIN__)
    if ( remove(backup) ) if ( errno != ENOENT ) {
      char *sys_err_msg = strerror(errno);
      if ( ! sys_err_msg ) sys_err_msg = "(unknown error)";
      iout << iERROR << "Error on removing file "
	<< backup << ": " << sys_err_msg << "\n" << endi;
      fflush(stdout);
    }
#endif
    while ( rename(filename,backup) )
    {
      if ( errno == EINTR ) continue;
      const char *sys_err_msg = strerror(errno);
      if ( ! sys_err_msg ) sys_err_msg = "(unknown error)";
      iout << iERROR << "Error on renaming file " << filename
	<< " to " << backup << ": " << sys_err_msg << "\n" << endi;
      fflush(stdout);
      if ( errno == EXDEV ) continue;
      break;
      // char errmsg[256];
      // sprintf(errmsg, "Error on renaming file %s to %s",filename,backup);
      // NAMD_err(errmsg);
    }
    delete [] backup;
  }
}

// same as open, only does error checking internally
int NAMD_open_text(const char *fname, int append) {
  int fd;

  //  open the file and die if the open fails
#ifdef WIN32
  while ( (fd = _open(fname, O_WRONLY|(append?O_APPEND:O_EXCL)|O_CREAT|O_TEXT,_S_IREAD|_S_IWRITE)) < 0) {
#else
#ifdef NAMD_NO_O_EXCL
  while ( (fd = open(fname, O_WRONLY|(append?O_APPEND:O_TRUNC)|O_CREAT,
#else
  while ( (fd = open(fname, O_WRONLY|(append?O_APPEND:O_EXCL)|O_CREAT,
#endif
                           S_IRUSR|S_IWUSR|S_IRGRP|S_IROTH)) < 0) {
#endif
    if ( errno != EINTR ) {
      char errmsg[1024];
      sprintf(errmsg, "Unable to open text file %s", fname);
      NAMD_err(errmsg);
    }
  }

  return fd;
}

// same as write, only does error checking internally
void NAMD_write(int fd, const char *buf, size_t count, const char *fname) {
  while ( count ) {
#if defined(WIN32) && !defined(__CYGWIN__)
    long retval = _write(fd,buf,count);
#else
    ssize_t retval = write(fd,buf,count);
#endif
    if ( retval < 0 && errno == EINTR ) retval = 0;
    if ( retval < 0 ) {
      char errmsg[1024];
      sprintf(errmsg, "Error on writing to file %s", fname);
      NAMD_err(errmsg);
    }
    if ( retval > count ) NAMD_bug("extra bytes written in NAMD_write()");
    buf += retval;
    count -= retval;
  }
}

// same as close, only does error checking internally
void NAMD_close(int fd, const char *fname) {
#ifdef WIN32
  while ( _close(fd) ) {
#else
  while ( close(fd) ) {
#endif
    if ( errno != EINTR ) {
      char errmsg[1024];
      sprintf(errmsg, "Error on closing file %s", fname);
      NAMD_err(errmsg);
    }
  }
}


/***************************************************************************
 Fopen(char *Filename, char *mode):  similar to fopen(filename,mode) except
 it checks for compressed file names too.
 For example:  Fopen("config");
   This will first look for the filename "config" (and return "r" file handle
   if it is found).
   Then it will look for "config.Z" (and run "zcat config.Z", returning
   a file handle to the uncompressed data if found).
   Then it will look for "config.gz" (and run "gzip -d -c config.gz", returning
   a file handle to the uncompressed data if found).
 ***************************************************************************/
FILE *Fopen	(const char *filename, const char *mode)
{
  struct stat buf;
  // check if basic filename exists (and not a directory)

#if defined(NOCOMPRESSED)
  if (!stat(filename,&buf))
    {
      FILE *rval;
      while ( ! (rval = fopen(filename,mode)) ) {
        if ( errno != EINTR ) break;
      }
      return(rval);
    }
#else
  if (!stat(filename,&buf))
    {
      if (!S_ISDIR(buf.st_mode)) {
        FILE *rval;
        while ( ! (rval = fopen(filename,mode)) ) {
          if ( errno != EINTR ) break;
        }
        return(rval);
      }
    }
  // check for a compressed file
  char *realfilename;
  char *command;
  FILE *fout;
  command = (char *)malloc(strlen(filename)+25);
  // check for .Z (unix compress)
  sprintf(command,"zcat %s.Z",filename);
  realfilename = command+5;
  iout << "Command = " << command << "\n" << endi;
  iout << "Filename.Z = " << realfilename << "\n" << endi;
  if (!stat(realfilename,&buf))
	{
	if (!S_ISDIR(buf.st_mode))
		{
		fout = popen(command,mode);
		// on HP-UX, the first character(s) out of pipe may be
		// garbage!  (Argh!)
		int C;
		do
		  {
		  C = fgetc(fout);
		  // iout << "C is " << C << "\n" << endi;
		  if (isalnum(C) || isspace(C))
			{
			ungetc(C,fout);
			C = -1;	// outta loop
			}
		  } while(C != -1);
		free(command);
		return(fout);
		}
	}
  // check for .gz (gzip)
  sprintf(command,"gzip -d -c %s.gz",filename);
  realfilename = command+11;
  iout << "Command = " << command << "\n" << endi;
  iout << "Filename.gz = " << realfilename << "\n" << endi;
  if (!stat(realfilename,&buf))
	{
	if (!S_ISDIR(buf.st_mode))
		{
		fout = popen(command,mode);
		// on HP-UX, the first character(s) out of pipe may be
		// garbage!  (Argh!)
		int C;
		do
		  {
		  C = fgetc(fout);
		  // iout << "C is " << C << "\n" << endi;
		  if (isalnum(C) || isspace(C))
			{
			ungetc(C,fout);
			C = -1;	// outta loop
			}
		  } while(C != -1);
		free(command);
		return(fout);
		}
	}
  free(command);
#endif /* !defined(NOCOMPRESSED) */

  return(NULL);
} /* Fopen() */

/***************************************************************************
 Fclose(FILE *fout):  similar to fclose(fout) except it first checks if the
 file handle fout is a named pipe.
 ***************************************************************************/
int	Fclose	(FILE *fout)
{
  int rc = -1;
#if !defined(NOCOMPRESSED)
  rc = pclose(fout);
#endif
  if (rc == -1)	// stream not associated with a popen()
    {
    rc = fclose(fout);
    }
  return rc;
} /* Fclose() */


