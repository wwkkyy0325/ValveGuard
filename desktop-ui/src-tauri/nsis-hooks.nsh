; ValveGuard NSIS hooks.
;
; Uninstalling asks whether the analysis history should be removed as well.
; Upgrades run the old uninstaller with _?= and keep the history without
; asking, so reinstalling never destroys records by accident.

Var ValveGuardDeleteUserData

!macro NSIS_HOOK_PREUNINSTALL
  StrCpy $ValveGuardDeleteUserData "0"

  ; Explicit opt-in for unattended uninstallers.
  ${GetOptions} $CMDLINE "/DELETEUSERDATA" $R1
  ${IfNot} ${Errors}
    StrCpy $ValveGuardDeleteUserData "1"
    Goto vg_uninstall_done
  ${EndIf}

  ; Silent uninstall keeps user data unless the flag above was given.
  IfSilent vg_uninstall_done

  ; Passive/update uninstalls are started by the installer, not by the
  ; user. Keep the records and skip the prompt.
  ${GetOptions} $CMDLINE "/UPDATE" $R1
  ${IfNot} ${Errors}
    Goto vg_uninstall_done
  ${EndIf}
  ${GetOptions} $CMDLINE "/P" $R1
  ${IfNot} ${Errors}
    Goto vg_uninstall_done
  ${EndIf}

  ; The bundled installer runs the old uninstaller with _?= during an
  ; upgrade or reinstall. That parameter comes first in the normal case.
  ${GetParameters} $R0
  StrCpy $R1 $R0 3
  ${If} $R1 == "_?="
    Goto vg_uninstall_done
  ${EndIf}

  MessageBox MB_YESNO|MB_ICONQUESTION \
    "Also delete ValveGuard analysis records?$\r$\n$\r$\nThis removes $DOCUMENTS\ValveGuard. Choose No to keep the record history for a later install." \
    IDNO vg_uninstall_done
  StrCpy $ValveGuardDeleteUserData "1"

  vg_uninstall_done:
!macroend

!macro NSIS_HOOK_POSTUNINSTALL
  ${If} $ValveGuardDeleteUserData == "1"
    RMDir /r "$DOCUMENTS\ValveGuard"
    RMDir /r "$LOCALAPPDATA\com.valveguard.review"
    RMDir /r "$APPDATA\com.valveguard.review"
  ${EndIf}
!macroend
