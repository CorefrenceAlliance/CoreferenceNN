# CoreferenceNN
Coreference with Deep Neural Networks

#### Building Project

1. Import the project into your preferred IDE
2. Download [jaws] bin
3. run command 'mvn install:install-file -Dfile=...\jaws-bin.jar -DgroupId=jaws -DartifactId=jaws -Dversion=1.2 -Dpackaging=jar'
4. from here, enter new info into maven
```
<properties>  
      ...  
      <jaws.version>1.2</jaws.version>  
      ...  
 </properties>  
 <dependencies>  
      ...  
      <dependency>  
           <groupId>jaws</groupId>  
           <artifactId>jaws</artifactId>  
           <version>${jaws.version}</version>  
      </dependency>  
      ...  
 </dependencies>  
```
5. Run Maven install on project.

> Run Maven install when pom.xml changes


[jaws]: http://lyle.smu.edu/~tspell/jaws/#downloads
