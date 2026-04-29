ThisBuild / evictionErrorLevel := Level.Info

resolvers += "jgit-repo" at "https://download.eclipse.org/jgit/maven"

addSbtPlugin("org.xerial.sbt" % "sbt-pack" % "0.9.3")
